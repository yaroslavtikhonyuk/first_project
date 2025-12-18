import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


class FieldVisualizer:
    """
    Класс для визуализации скалярных и векторных полей
    """

    def __init__(self, x_range, y_range, grid_points):
        """
        Инициализация визуализатора полей

        Parameters:
        -----------
        x_range : tuple
            Диапазон значений по оси X (min, max)
        y_range : tuple
            Диапазон значений по оси Y (min, max)
        grid_points : int
            Количество точек на сетке в каждом направлении
        """
        self.x_range = x_range
        self.y_range = y_range
        self.grid_points = grid_points

        # Тип поля: '2D_scalar', '2D_vector', '3D_scalar', '3D_vector'
        self.field_type = None

        # Создание координатной сетки
        self.x = np.linspace(x_range[0], x_range[1], grid_points)
        self.y = np.linspace(y_range[0], y_range[1], grid_points)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def generate_scalar_field(self, scalar_func):
        """
        Генерация данных скалярного поля

        Parameters:
        -----------
        scalar_func : function
            Функция f(x, y), возвращающая скалярное значение

        Note:
        -----
        Результат сохраняется в переменную класса:
        self.Z
        """
        # Установка типа поля
        self.field_type = '2D_scalar'
        
        # Вычисление значений скалярного поля
        self.Z = scalar_func(self.X, self.Y)

    def plot_scalar_field(self,
                          cmap='viridis', show_contours=True, show_colorbar=True):
        """
        Визуализация скалярного поля

        Parameters:
        -----------
        cmap : str
            Цветовая карта
        show_contours : bool
            Показывать ли линии уровня
        show_colorbar : bool
            Показывать ли цветовую шкалу

        Note:
        -----
        Использует данные из переменной класса:
        self.Z
        """
        Z = self.Z

        # Создание графика
        fig, ax = plt.subplots(figsize=(10, 8))

        # Визуализация поля
        im = ax.imshow(Z, extent=[self.x_range[0], self.x_range[1],
                                  self.y_range[0], self.y_range[1]],
                       origin='lower', cmap=cmap, aspect='auto')

        # Добавление линий уровня
        if show_contours:
            contours = ax.contour(self.X, self.Y, Z, levels=15,
                                  colors='black', linewidths=0.5, alpha=0.7)
            #ax.clabel(contours, inline=True, fontsize=8)

        # Настройка графика
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)

        ax.grid(True, alpha=0.3)

        # Добавление цветовой шкалы
        if show_colorbar:
            plt.colorbar(im, ax=ax, label='Значение функции')

        plt.tight_layout()
        plt.show()

        return Z


    def generate_vector_field(self, vector_func, scale1=100):
        """
        Генерация данных векторного поля

        Parameters:
        -----------
        vector_func : function
            Функция F(x, y), возвращающая кортеж (X1, Y1) - компоненты вектора
        scale1 : float
            Масштаб для нормализованных векторов

        Note:
        -----
        Результат сохраняется в переменные класса:
        self.X1, self.Y1, self.X1_norm, self.Y1_norm, self.magnitude
        """
        # Установка типа поля
        self.field_type = '2D_vector'
        
        # Вычисление компонент векторного поля
        X1, Y1 = vector_func(self.X, self.Y)

        # Вычисление длины векторов
        magnitude = np.sqrt(X1 ** 2 + Y1 ** 2)

        # Нормализация векторов для одинаковой длины
        X1_norm = X1 / (magnitude + 1e-10) * scale1
        Y1_norm = Y1 / (magnitude + 1e-10) * scale1

        # Сохранение данных в переменные класса
        self.X1 = X1
        self.Y1 = Y1
        self.X1_norm = X1_norm
        self.Y1_norm = Y1_norm
        self.magnitude = magnitude

    def transform_vector_field(self, transform_func, scale1=100):
        """
        Изменение векторного поля по функции от координат и значений поля

        Parameters:
        -----------
        transform_func : function
            Функция F(X, Y, X1, Y1), принимающая координаты и текущие значения поля,
            возвращающая кортеж (new_X1, new_Y1) - новые компоненты вектора
        scale1 : float
            Масштаб для нормализованных векторов

        Note:
        -----
        Обновляет переменные класса:
        self.X1, self.Y1, self.X1_norm, self.Y1_norm, self.magnitude
        """
        # Вычисление новых компонент векторного поля
        X1_new, Y1_new = transform_func(self.X, self.Y, self.X1, self.Y1)

        # Вычисление длины векторов
        magnitude = np.sqrt(X1_new ** 2 + Y1_new ** 2)

        # Нормализация векторов для одинаковой длины
        X1_norm = X1_new / (magnitude + 1e-10) * scale1
        Y1_norm = Y1_new / (magnitude + 1e-10) * scale1

        # Обновление данных в переменных класса
        self.X1 = X1_new
        self.Y1 = Y1_new
        self.X1_norm = X1_norm
        self.Y1_norm = Y1_norm
        self.magnitude = magnitude

    def plot_vector_field(self,
                          color_map='plasma', scale=30, width=0.005,
                          show_magnitude_contour=False):
        """
        Отрисовка векторного поля

        Parameters:
        -----------
        color_map : str
            Цветовая карта для отображения длины векторов
        scale : float
            Масштабирование длины векторов
        width : float
            Толщина векторов
        show_magnitude_contour : bool
            Показывать ли изолинии длины векторов

        Note:
        -----
        Использует данные из переменных класса:
        self.X1_norm, self.Y1_norm, self.magnitude
        """
        X1_norm = self.X1_norm
        Y1_norm = self.Y1_norm
        magnitude = self.magnitude

        # Создание графика
        fig, ax = plt.subplots(figsize=(12, 9))

        # Отображение изолиний длины (опционально)
        if show_magnitude_contour:
            contour = ax.contourf(self.X, self.Y, magnitude,
                                  levels=20, alpha=0.3, cmap=color_map + '_r')
            plt.colorbar(contour, ax=ax, label='Длина вектора')

        # Создание цветовой карты
        cmap = plt.cm.get_cmap(color_map)

        # Нормализация для цветов
        if magnitude.max() > magnitude.min():
            norm = plt.Normalize(vmin=magnitude.min(), vmax=magnitude.max())
            colors = cmap(norm(magnitude.flatten()))
        else:
            norm = plt.Normalize(vmin=0, vmax=1)
            colors = cmap(0.5 * np.ones_like(magnitude.flatten()))

        # Визуализация векторного поля
        quiver = ax.quiver(self.X, self.Y, X1_norm, Y1_norm,
                           color=colors,
                           angles='xy',
                           scale_units='xy',
                           scale=scale,
                           width=width,
                           alpha=0.8)

        # Добавление colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Магнитуда вектора', fontsize=12)

        # Настройка графика
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')

        ax.set_title('Векторное поле\n(длина векторов одинакова, цвет отражает фактическую длину)',
                     fontsize=14, pad=20)

        plt.tight_layout()
        plt.show()


# Примеры функций для демонстрации
def example_scalar_func_1(x, y):
    """Пример 1: Параболоид"""
    return (x - 50) ** 2 + (y - 50) ** 2


def example_scalar_func_2(x, y):
    """Пример 2: Синусоидальная функция"""
    return np.gradient(example_scalar_func_1(x, y), x, axis=0) + np.gradient(example_scalar_func_1(x, y), y, axis=1)


def example_scalar_func_3(x, y):
    """Пример 3: Функция с седловой точкой"""
    return x ** 2 - y ** 2


def example_vector_func_1(x, y):
    """Пример 1: Векторное поле, направленное к центру"""
    U = -x
    V = -y
    return U, V


def example_vector_func_2(x, y):
    """Пример 2: Вращательное поле"""
    U = -y
    V = x
    return U, V


def example_vector_func_3(x, y):
    """Пример 3: Градиент параболоида"""
    U, V = np.gradient(example_scalar_func_1(x, y), x, y)
    return U, V


def example_vector_func_4(x, y):
    """Пример 4: Сложное векторное поле"""
    U, V = np.curl(example_vector_func_2(x, y))
    return U, V



field = FieldVisualizer((0, 100), (0, 100), 100)
field.generate_scalar_field(example_scalar_func_1)
field = FieldVisualizer((0, 100), (0, 100), 100)
field.generate_scalar_field(example_scalar_func_2)
field.plot_scalar_field(cmap='viridis', show_contours=True, show_colorbar=True)
field1 = FieldVisualizer((0, 100), (0, 100), 10)
field1.generate_vector_field(example_vector_func_2, scale1=150)
field1.plot_vector_field(color_map='plasma', scale=30, width=0.005,
                         show_magnitude_contour=False)
field1.generate_vector_field(example_vector_func_3, scale1=150)
field1.plot_vector_field(color_map='plasma', scale=30, width=0.005,
                         show_magnitude_contour=False)
field1.generate_vector_field(example_vector_func_4, scale1=150)
field1.plot_vector_field(color_map='plasma', scale=30, width=0.005,
                         show_magnitude_contour=False)