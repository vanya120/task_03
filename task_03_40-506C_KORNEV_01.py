import numpy
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import tools

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    #Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 500

    #Размер области моделирования в метрах
    X = 0.5

    #Размер ячейки разбиения
    dx = 1e-3

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = 50
    
    # Датчики для регистрации поля
    probesPos = [100]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    #Гауссов импульс
    A0 = 100        #Ослабление сигнала в момент времени t=0
    Amax = 100      #Уровень ослабления спектра сигнала на частоте Fmax
    Fmax = 3e9      #Максимальная частота в спектре сигнала
    Wg = numpy.sqrt(numpy.log(Amax)) / (numpy.pi * Fmax)
    Dg = Wg * numpy.sqrt(numpy.log(A0))
    Nwg = Wg / dt
    Ndg = Dg / dt
    
    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize)

    for probe in probes:
        probe.addData(Ez, Hy)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)
    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for t in range(1, maxTime):
        # Граничные условия для поля H
        Hy[-1] = Hy[-2]

        # Расчет компоненты поля H
        Ez_shift = Ez[1:]
        Hy[:-1] = Hy[:-1] + (Ez_shift - Ez[:-1]) * Sc / W0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= (Sc / W0) * numpy.exp(-((t - Ndg - sourcePos) / Nwg) ** 2 )

        # Граничные условия для поля E
        Ez[0] = Ez[1]

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:] = Ez[1:] + (Hy[1:] - Hy_shift) * Sc * W0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += Sc * numpy.exp(-(((t + 0.5) - (sourcePos - 0.5) - Ndg) / Nwg) ** 2 )

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % 5 == 0:
            display.updateData(display_field, t)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    #Спектр сигнала в датчике
    #Размер массива
    size = 4096

    #Шаг по частоте
    df = 1.0 / (size * dt)
    
    #Расчет спектра
    spectrum = numpy.abs(fft(probe.E, size))
    spectrum = fftshift(spectrum)
    f = numpy.arange(-size / 2 * df, size / 2 * df, df)

    #Построение графика
    plt.plot(f, spectrum /numpy.max(spectrum))
    plt.grid
    plt.xlim(0, 5e9)
    plt.xlabel('Частота, Гц')
    plt.ylabel(r'$\frac{|S|}{S_{max}}$')
    plt.grid()
    plt.show()
    
