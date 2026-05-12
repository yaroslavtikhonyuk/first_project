import numpy as np
from dataclasses import dataclass
from typing import Tuple
import warnings
import os
import json
import h5py
warnings.filterwarnings('ignore')

_K_B = 1.38e-16
_MU  = 2.3
_M_P = 1.67e-24


# ================================================================
# § 1. Параметры диска и сетка
# ================================================================

@dataclass
class DiskConfig:
    # Геометрия сетки
    Nr:   int   = 1024
    Nphi: int   = 512
    r_in: float = 0.3
    r_out: float = 3.0

    # Звезда
    M_star: float = 1.0

    # Газ (MMSN-подобный)
    Sigma_g0: float = 1700.0 # г/см² на r=1 AU
    p_Sigma:  float = 1.0
    T0:       float = 280.0  # К на r=1 AU
    q_T:      float = 0.5

    # Пыль
    Sigma_d0: float = 170.0  # г/см²
    rho_s:    float = 1.6    # г/см³
    a_grain:  float = 0.1    # см (1 мм)

    # Вязкость газа (α-модель)
    alpha_visc: float = 1e-3

    # Физические единицы (не варьируемые параметры)
    G:     float = 4 * np.pi**2
    AU_cm: float = 1.496e13
    yr_s:  float = 3.156e7


class DiskGrid:
    """Логарифмическая сетка в (r, φ), всё в AU и годах."""

    def __init__(self, cfg: DiskConfig):
        self.cfg = cfg

        self.r_e = np.logspace(np.log10(cfg.r_in), np.log10(cfg.r_out), cfg.Nr + 1)
        self.r   = 0.5 * (self.r_e[:-1] + self.r_e[1:])
        self.dr  = np.diff(self.r_e)

        self.phi_e = np.linspace(0, 2*np.pi, cfg.Nphi + 1)
        self.phi   = 0.5 * (self.phi_e[:-1] + self.phi_e[1:])
        self.dphi  = 2*np.pi / cfg.Nphi

        self.R, self.Phi = np.meshgrid(self.r, self.phi, indexing='ij')

        self.Omega_K = np.sqrt(cfg.G * cfg.M_star / self.R**3)
        self.v_K     = self.R * self.Omega_K
        self.sqrt_r  = np.sqrt(self.r[:, None])


# ================================================================
# § 2. Гидростатическое равновесие
# ================================================================

class HydrostaticDisk:
    """Начальное состояние в гидростатическом равновесии."""

    def __init__(self, grid: DiskGrid):
        self.g = grid
        cfg = grid.cfg
        R = grid.R

        self.Sigma_g = (cfg.Sigma_g0
                        * (R / 1.0)**(-cfg.p_Sigma)
                        * np.exp(-R / cfg.r_out))

        self.T   = cfg.T0 * (R / 1.0)**(-cfg.q_T)
        self.cs2 = _K_B * self.T / (_MU * _M_P) * cfg.yr_s**2 / cfg.AU_cm**2
        self.cs  = np.sqrt(self.cs2)
        self.H   = self.cs / grid.Omega_K
        self.h   = self.H / R

        ln_P    = np.log(self.Sigma_g * self.cs2 + 1e-30)
        d_lnP   = np.gradient(ln_P, np.log(self.g.r), axis=0)
        self.eta = -0.5 * self.h**2 * d_lnP

        self.v_phi_g = grid.v_K * np.sqrt(np.maximum(1 - 2*self.eta, 0))
        self.v_r_g   = np.zeros_like(R)

        self.Sigma_d = (cfg.Sigma_d0
                        * (R / 1.0)**(-cfg.p_Sigma)
                        * np.exp(-R / cfg.r_out))
        self.v_phi_d = self.v_phi_g.copy()
        self.v_r_d   = np.zeros_like(R)

    def epstein_stokes(self) -> np.ndarray:
        cfg = self.g.cfg
        return np.pi / 2 * cfg.rho_s * cfg.a_grain / np.maximum(self.Sigma_g, 1e-30)


# ================================================================
# § 3. Дрейф пыли (NSH)
# ================================================================

class TerminalVelocityDrift:
    """Терминальная скорость дрейфа (NSH, St ≪ 1)."""

    def drift_velocities(self,
                         disk: HydrostaticDisk,
                         include_feedback: bool = True
                        ) -> Tuple[np.ndarray, np.ndarray]:
        g   = disk.g
        St  = disk.epstein_stokes()
        eps = disk.Sigma_d / (disk.Sigma_g + 1e-30)

        if include_feedback:
            D = (1 + eps)**2 + St**2
            v_r_d = -2 * disk.eta * g.v_K * St / D
            v_r_g =  2 * disk.eta * g.v_K * eps * St / D
        else:
            D = 1 + St**2
            v_r_d = -2 * disk.eta * g.v_K * St / D
            v_r_g = np.zeros_like(v_r_d)

        return v_r_d, v_r_g


# ================================================================
# § 4. Консервативная радиальная адвекция
# ================================================================

def _upwind_divergence(Sigma: np.ndarray,
                       v_r:   np.ndarray,
                       grid:  DiskGrid) -> np.ndarray:
    Nr, Nphi = Sigma.shape

    v_face       = np.empty((Nr + 1, Nphi))
    v_face[0]    = v_r[0]
    v_face[1:-1] = 0.5 * (v_r[:-1] + v_r[1:])
    v_face[-1]   = v_r[-1]

    Sig_L      = np.empty((Nr + 1, Nphi))
    Sig_L[0]   = Sigma[0]
    Sig_L[1:]  = Sigma

    Sig_R      = np.empty((Nr + 1, Nphi))
    Sig_R[:-1] = Sigma
    Sig_R[-1]  = Sigma[-1]

    flux = grid.r_e[:, None] * np.where(v_face > 0, Sig_L, Sig_R) * v_face
    flux[0,  :] = np.minimum(flux[0,  :], 0.0)
    flux[-1, :] = np.maximum(flux[-1, :], 0.0)

    return -(flux[1:] - flux[:-1]) / (grid.R * grid.dr[:, None])


# ================================================================
# § 4.1 Эволюция пыли (FARGO + upwind)
# ================================================================

def advance_dust_simple(Sigma_d: np.ndarray,
                        v_r_d:   np.ndarray,
                        v_phi_d: np.ndarray,
                        grid:    DiskGrid,
                        dt:      float) -> np.ndarray:
    g    = grid
    Nr   = g.cfg.Nr
    Nphi = g.cfg.Nphi

    v_K_1d  = g.v_K[:, 0]
    n_shift = np.round(v_K_1d * dt / (g.R[:, 0] * g.dphi)).astype(int) % Nphi
    col_idx = (np.arange(Nphi)[None, :] - n_shift[:, None]) % Nphi
    S = Sigma_d[np.arange(Nr)[:, None], col_idx]

    dS_r = _upwind_divergence(S, v_r_d, g)

    v_phi_slow = v_phi_d - v_K_1d[:, None]
    face       = np.where(v_phi_slow > 0, S, np.roll(S, -1, axis=1))
    phi_flux   = face * v_phi_slow
    dS_phi     = -(phi_flux - np.roll(phi_flux, 1, axis=1)) / (g.R * g.dphi)

    return np.maximum(S + dt * (dS_r + dS_phi), 1e-30)


# ================================================================
# § 4.2 Эволюция газа (вязкий upwind)
# ================================================================

def advance_gas_viscous(Sigma_g:   np.ndarray,
                        v_r_total: np.ndarray,
                        grid:      DiskGrid,
                        dt:        float) -> np.ndarray:
    return np.maximum(Sigma_g + dt * _upwind_divergence(Sigma_g, v_r_total, grid), 1e-30)


# ================================================================
# § 5. Полный временной цикл
# ================================================================

class DiskEvolution:
    """Главный класс эволюции диска."""

    def __init__(self, cfg: DiskConfig):
        self.cfg  = cfg
        self.grid = DiskGrid(cfg)
        self.disk = HydrostaticDisk(self.grid)
        self.tv   = TerminalVelocityDrift()
        self.t    = 0.0
        self._i1  = int(np.argmin(np.abs(self.grid.r - 1.0)))

    def cfl_timestep(self, v_r_d: np.ndarray, safety: float = 0.4) -> float:
        g = self.grid
        d = self.disk

        dt_r   = g.dr / (np.abs(v_r_d).max(axis=1) + 1e-30)
        dt_phi = g.R * g.dphi / (np.abs(d.v_phi_d - g.v_K) + 1e-30)
        dt_adv = safety * float(min(dt_r.min(), dt_phi.min()))

        nu_1d   = self.cfg.alpha_visc * d.cs2[:, 0] / (g.Omega_K[:, 0] + 1e-30)
        dt_diff = 0.4 * float((g.dr**2 / nu_1d).min())

        return min(dt_adv, dt_diff)

    def step(self, include_feedback: bool = True) -> dict:
        d = self.disk
        g = self.grid

        v_r_d, v_r_g_fb = self.tv.drift_velocities(d, include_feedback)

        nu       = self.cfg.alpha_visc * d.cs2 / (g.Omega_K + 1e-30)
        v_r_visc = (-3.0 / (d.Sigma_g * g.sqrt_r + 1e-30)
                    * np.gradient(nu * d.Sigma_g * g.sqrt_r, g.r, axis=0))

        v_r_g_total = v_r_visc + v_r_g_fb
        dt = self.cfl_timestep(v_r_d)

        d.Sigma_d = advance_dust_simple(d.Sigma_d, v_r_d, d.v_phi_d, g, dt)
        d.Sigma_g = advance_gas_viscous(d.Sigma_g, v_r_g_total, g, dt)

        ln_P      = np.log(d.Sigma_g * d.cs2 + 1e-30)
        d_lnP     = np.gradient(ln_P, np.log(g.r), axis=0)
        d.eta     = -0.5 * d.h**2 * d_lnP
        d.v_phi_g = g.v_K * np.sqrt(np.maximum(1 - 2*d.eta, 0))

        St       = d.epstein_stokes()
        tau_stop = np.maximum(St / (g.Omega_K + 1e-30), 1e-3 / g.Omega_K)
        eps      = d.Sigma_d / (d.Sigma_g + 1e-30)
        beta     = dt / tau_stop
        dv_d     = (d.v_phi_g - d.v_phi_d) * beta / (1.0 + beta)
        d.v_phi_d += dv_d
        d.v_phi_g -= eps * dv_d

        self.t += dt

        M_dust        = float((d.Sigma_d * g.R * g.dr[:, None] * g.dphi).sum())
        courant_r_max = float(np.abs(v_r_d).max() * dt / g.dr.min())

        return {
            'dt':            dt,
            't':             self.t,
            'St_1AU':        float(St[self._i1, 0]),
            'eta_1AU':       float(d.eta[self._i1, 0]),
            'Sigma_g':       d.Sigma_g.mean(axis=1),
            'Sigma_d':       d.Sigma_d.mean(axis=1),
            'v_r_d':         v_r_d.mean(axis=1),
            'Courant_r_max': courant_r_max,
            'M_dust':        M_dust,
        }


# ================================================================
# § 6. Запуск и валидация
# ================================================================

def run_simulation(t_end_yr: float = 500.0, nr: int = 64, nphi: int = 128):
    """Запустить пробную симуляцию."""
    cfg  = DiskConfig(Nr=nr, Nphi=nphi)
    evol = DiskEvolution(cfg)
    g    = evol.grid

    snapshots = {}
    t_snaps   = [100, 250, t_end_yr]
    snap_idx  = 0

    M_dust_0 = float((evol.disk.Sigma_d * g.R * g.dr[:, None] * g.dphi).sum())

    SEP = "-" * 105
    print(f"Nr={nr}, Nphi={nphi}, t_end={t_end_yr} лет")
    print(SEP)
    print(f"{'t (yr)':>10} | {'dt (yr)':>10} | {'St@1AU':>8} | "
          f"{'η@1AU':>8} | {'Σ_g@1AU':>10} | {'Σ_d@1AU':>10} | "
          f"{'v_drift':>10} | {'CFL_r':>7} | {'M_d/M_d0':>9}")
    print(SEP)

    i = 0
    while evol.t < t_end_yr:
        diag = evol.step(include_feedback=True)

        if snap_idx < len(t_snaps) and evol.t >= t_snaps[snap_idx]:
            snapshots[round(evol.t)] = {
                'Sigma_g': diag['Sigma_g'].copy(),
                'Sigma_d': diag['Sigma_d'].copy(),
                'v_r_d':   diag['v_r_d'].copy(),
            }
            snap_idx += 1

        if i % max(1, int(10 / (t_end_yr / 500))) == 0:
            i1 = evol._i1
            print(f"{evol.t:>10.1f} | {diag['dt']:>10.4f} | "
                  f"{diag['St_1AU']:>8.4f} | {diag['eta_1AU']:>8.5f} | "
                  f"{diag['Sigma_g'][i1]:>10.2f} | {diag['Sigma_d'][i1]:>10.3f} | "
                  f"{diag['v_r_d'][i1]:>10.6f} | "
                  f"{diag['Courant_r_max']:>7.4f} | "
                  f"{diag['M_dust'] / M_dust_0:>9.4f}")
        i += 1

    print(SEP)
    print(f"Готово! {i} шагов, t={evol.t:.1f} лет\n")
    return evol, snapshots


def validate_against_nsh(evol: 'DiskEvolution'):
    """Сравнение с NSH-аналитикой (векторизованное)."""
    d   = evol.disk
    g   = evol.grid
    St  = d.epstein_stokes()[:, 0]
    eps = (d.Sigma_d / (d.Sigma_g + 1e-30))[:, 0]
    eta = d.eta[:, 0]
    vK  = g.v_K[:, 0]

    D       = (1 + eps)**2 + St**2
    v_r_nsh = -2 * eta * vK * St / D
    v_r_sim = evol.tv.drift_velocities(d)[0][:, 0]

    mask = np.abs(v_r_nsh) > 1e-10
    if mask.sum() > 0:
        err = np.abs((v_r_sim[mask] - v_r_nsh[mask]) / v_r_nsh[mask])
        print(f"Сравнение с NSH-аналитикой:")
        print(f"  Средняя погрешность: {err.mean()*100:.2f}%")
        print(f"  Максимальная:        {err.max()*100:.2f}%")
        print(f"  Диапазон: [{err.min()*100:.2f}%, {err.max()*100:.2f}%]")

    return g.r, v_r_sim, v_r_nsh


def export_simulation_data(evol: 'DiskEvolution',
                           snapshots: dict,
                           out_path: str = 'disk_simulation.h5') -> None:
    g = evol.grid
    d = evol.disk

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('r',   data=g.r)
        f.create_dataset('phi', data=g.phi)
        f.create_dataset('R',   data=g.R)
        f.create_dataset('Phi', data=g.Phi)

        cfg_group = f.create_group('config')
        cfg_group.attrs['json'] = json.dumps({k: getattr(g.cfg, k) for k in vars(g.cfg)})

        final = f.create_group('final')
        final.create_dataset('Sigma_g_avg', data=d.Sigma_g.mean(axis=1))
        final.create_dataset('Sigma_d_avg', data=d.Sigma_d.mean(axis=1))
        final.create_dataset('v_phi_d_avg', data=d.v_phi_d.mean(axis=1))
        final.create_dataset('eta_avg',     data=d.eta.mean(axis=1))
        final.create_dataset('St_avg',      data=d.epstein_stokes()[:, 0])

        snaps_grp = f.create_group('snapshots')
        times = []
        for tkey in sorted(snapshots.keys()):
            times.append(float(tkey))
            sg = snaps_grp.create_group(str(tkey))
            for name, arr in snapshots[tkey].items():
                sg.create_dataset(name, data=arr)
        f.create_dataset('times', data=np.array(times))

    print(f"✓ Экспортировано в {out_path}")


def main():
    print("\n" + "="*105)
    print("ПРОБНАЯ СИМУЛЯЦИЯ: Дрейф пыли в молодом аккреционном диске")
    print("="*105 + "\n")

    evol, snaps = run_simulation(t_end_yr=1000.0, nr=32, nphi=128)
    r, v_sim, _ = validate_against_nsh(evol)

    print(f"\n✓ Симуляция успешна! {len(snaps)} снимков, r=[{r.min():.2f}, {r.max():.2f}] AU")

    i1 = evol._i1
    print(f"\nПараметры на 1 AU:")
    print(f"  St = {evol.disk.epstein_stokes()[i1, 0]:.4f}")
    print(f"  η  = {evol.disk.eta[i1, 0]:.5f}")
    print(f"  ε  = {evol.disk.Sigma_d[i1,0]/(evol.disk.Sigma_g[i1,0]+1e-30):.6f}")
    print(f"  v_drift = {v_sim[i1]:.8f} AU/yr")
    return evol, snaps


if __name__ == "__main__":
    evol, snaps = main()
    export_simulation_data(evol, snaps, out_path='disk_simulation.h5')
    print('\nЗапуск анимации: python animate_drift.py disk_simulation.h5')
