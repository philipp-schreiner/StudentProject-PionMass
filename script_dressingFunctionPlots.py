import matplotlib.pyplot as plt
import numpy as np

from Classes import DressingFunctions
from Functions import plot_dressing_contour

N = 150

A_init = np.ones((1,N))
B_init = 0.01*np.ones((1,N))

# Toggle to generate different plots
plot_different_choices_of_integration_method = False
plot_difference_insertion_extrapolation = False
plot_timelike_dressing = False
plot_complex_plane = False
check_Literature_numbers = False

if plot_different_choices_of_integration_method:

    dressing1 = DressingFunctions(A_init,B_init,D=16,omega=0.5,m=0)
    A1, B1, p_squared1 = dressing1.solve_laguerre(n=N, res=1e-15)

    dressing2 = DressingFunctions(A_init,B_init,D=16,omega=0.5,m=0)
    A2, B2, p_squared2 = dressing2.solve_chebyshev_log(n=N,type='1st',res=1e-15)

    dressing3 = DressingFunctions(A_init,B_init,D=16,omega=0.5,m=0)
    A3, B3, p_squared3 = dressing3.solve_chebyshev(n=N,type='1st', res=1e-15)

    fig, ax = plt.subplots(1,3,figsize=(13,5))
    ax[0].plot(p_squared1.T, A1.T, 'x', label='$A(p^2)$')
    ax[0].plot(p_squared1.T, B1.T, 'x', label='$B(p^2)$')
    ax[0].plot(p_squared1.T, B1.T/A1.T, 'x', label='$M(p^2)$')
    ax[0].set_xscale('log')
    ax[0].grid(True, which='both')
    ax[0].set_ylim((0,1.5))
    ax[0].set_xlim((1e-5,1e4))
    ax[0].set_xlabel('$p^2~/~GeV$')
    ax[0].set_title('Gauss-Laguerre')
    ax[0].legend()

    ax[1].plot(p_squared2.T, A2.T, 'x', label='$A(p^2)$')
    ax[1].plot(p_squared2.T, B2.T, 'x', label='$B(p^2)$')
    ax[1].plot(p_squared2.T, B2.T/A2.T, 'x', label='$M(p^2)$')
    ax[1].set_xscale('log')
    ax[1].grid(True, which='both')
    ax[1].set_ylim((0,1.5))
    ax[1].set_xlim((1e-5,1e4))
    ax[1].set_xlabel('$p^2~/~GeV$')
    ax[1].set_title('Gauss-Chebyshev (log.Sub.)')
    ax[1].legend()

    ax[2].plot(p_squared3.T, A3.T, 'x', label='$A(p^2)$')
    ax[2].plot(p_squared3.T, B3.T, 'x', label='$B(p^2)$')
    ax[2].plot(p_squared3.T, B3.T/A3.T, 'x', label='$M(p^2)$')
    ax[2].set_xscale('log')
    ax[2].grid(True, which='both')
    ax[2].set_ylim((0,1.5))
    ax[2].set_xlim((1e-5,1e4))
    ax[2].set_xlabel('$p^2~/~GeV$')
    ax[2].set_title('Gauss-Chebyshev (frac.Sub.)')
    ax[2].legend()

    #plt.savefig(f'graphics/integrationMethodComparison.pdf', bbox_inches='tight')
    plt.show()

if plot_difference_insertion_extrapolation:
    meta = {'D': 16,
            'omega': 0.5,
            'm': 0,
            'N_nodes': N,
            'chebyshev_type': '1st',
            'residuum': 1e-15,
            'M_chebyshev_int': 7}

    dressing = DressingFunctions(A_init, B_init,
                                 D=meta['D'], omega=meta['omega'], m=meta['m'])
    A, B, p_squared = dressing.solve_chebyshev(n=N,
                                 type=meta['chebyshev_type'],
                                 res=meta['residuum'])

    x = np.logspace(-5,5,1000)
    insertA, insertB = dressing.AB_fun(x)
    extraA, extraB = dressing.AB_fun_int(M=meta['M_chebyshev_int'])

    fig, ax = plt.subplots(1,2,figsize=(13,5))

    ax[0].plot(x.T, insertA.T, label='$A(p^2)$')
    ax[0].plot(x.T, insertB.T, '--', label='$B(p^2)$')
    ax[0].plot(x.T, insertB.T/insertA.T, '--', label='$M(p^2)$')
    ax[0].set_xscale('log')
    ax[0].grid(True, which='both')
    ax[0].set_ylim((0,1.5))
    ax[0].set_xlabel('$p^2~/~GeV$')
    ax[0].set_title('Simple Insertion')
    ax[0].legend()

    ax[1].plot(x, extraA(x), label='$A(p^2)$')
    ax[1].plot(x, extraB(x), '--', label='$B(p^2)$')
    ax[1].plot(x, extraB(x)/extraA(x), '--', label='$M(p^2)$')
    ax[1].set_xscale('log')
    ax[1].grid(True, which='both')
    ax[1].set_ylim((0,1.5))
    ax[1].set_xlabel('$p^2~/~GeV$')
    ax[1].set_title(f'Chebyshev Extrapolation (N={meta["M_chebyshev_int"]})')
    ax[1].legend()

    #plt.savefig(f'graphics/extrapolationMethodComparison.pdf', bbox_inches='tight')
    plt.show()

if plot_timelike_dressing:
    dressing1 = DressingFunctions(A_init, B_init, D=16, omega=0.5, m=0)
    dressing1.solve_chebyshev(n=N, type='1st', res=1e-15)
    extraA1, extraB1 = dressing1.AB_fun_int(M=7)

    dressing2 = DressingFunctions(A_init, B_init, D=16, omega=0.5, m=0.115)
    dressing2.solve_chebyshev(n=N, type='1st', res=1e-15)
    extraA2, extraB2 = dressing2.AB_fun_int(M=7)

    x = np.logspace(-5,5,1000)
    x_timelike = np.linspace(-10,0,1000)

    fig, ax = plt.subplots(2,2,figsize=(13,8))
    ax[0,0].plot(x, extraA1(x), label='$A(p^2)$')
    ax[0,0].plot(x, extraB1(x), '--', label='$B(p^2)$')
    ax[0,0].plot(x, extraB1(x)/extraA1(x), '--', label='$M(p^2)$')
    ax[0,0].set_xscale('log')
    ax[0,0].grid(True, which='both')
    ax[0,0].set_ylim((0,1.5))
    ax[0,0].set_xlabel('$p^2~/~GeV$')
    ax[0,0].legend()

    ax[1,0].plot(x, extraA2(x), label='$A(p^2)$')
    ax[1,0].plot(x, extraB2(x), '--', label='$B(p^2)$')
    ax[1,0].plot(x, extraB2(x)/extraA2(x), '--', label='$M(p^2)$')
    ax[1,0].set_xscale('log')
    ax[1,0].grid(True, which='both')
    ax[1,0].set_ylim((0,1.5))
    ax[1,0].set_xlabel('$p^2~/~GeV$')
    ax[1,0].legend()

    ax[0,1].plot(x_timelike, (extraB1(x_timelike)/extraA1(x_timelike))**2, label='$M^2(p^2)$',color=ax[0,0].lines[2].get_color())
    ax[0,1].grid(True, which='both')
    ax[0,1].set_ylim((0,10))
    ax[0,1].set_xlabel('$p^2~/~GeV$')
    ax[0,1].set_title('')
    ax[0,1].legend()

    ax[1,1].plot(x_timelike, (extraB2(x_timelike)/extraA2(x_timelike))**2, label='$M^2(p^2)$',color=ax[0,0].lines[2].get_color())
    ax[1,1].grid(True, which='both')
    ax[1,1].set_ylim((0,10))
    ax[1,1].set_xlabel('$p^2~/~GeV$')
    ax[1,1].set_title('')
    ax[1,1].legend()

    #plt.savefig(f'graphics/differentTLMasses_plusTimelike.pdf', bbox_inches='tight')
    plt.show()

if plot_complex_plane:
    dressing1 = DressingFunctions(A_init, B_init, D=16, omega=0.5, m=0)
    dressing1.solve_chebyshev(n=N, type='1st', res=1e-15)
    extraA1, extraB1 = dressing1.AB_fun_int(M=7)

    dressing2 = DressingFunctions(A_init, B_init, D=16, omega=0.5, m=0.115)
    dressing2.solve_chebyshev(n=N, type='1st', res=1e-15)
    extraA2, extraB2 = dressing2.AB_fun_int(M=7)

    gridspec = dict(wspace=0, hspace=0, width_ratios=[1, 1, 0.2, 1, 1], height_ratios=[1, 1, 0.3, 1, 1])
    fig, ax = plt.subplots(5, 5, figsize=(13,9), gridspec_kw=gridspec)

    for i in range(5):
        ax[2,i].set_visible(False)
        ax[i,2].set_visible(False)

    levels = np.linspace(0,1.5,20)

    # Global view
    xDom_bottom = (-0.07,1e-2,1e3)
    yDom_bottom = (-0.25,0.25)
    CS = plot_dressing_contour (extraA1, eta=1/2, M=extraB1(0)/10,
                           xDom=xDom_bottom, yDom=yDom_bottom,
                           axs=[ax[3,0],ax[3,1]], n_mesh=100,
                           abs_max=2, levels=levels)
    plot_dressing_contour (extraA2, eta=1/2, M=extraB2(0)/10,
                          xDom=xDom_bottom, yDom=yDom_bottom,
                          axs=[ax[4,0],ax[4,1]], n_mesh=100,
                          abs_max=2, levels=levels)

    plot_dressing_contour (extraB1, eta=1/2, M=extraB1(0)/10,
                         xDom=xDom_bottom, yDom=yDom_bottom,
                         axs=[ax[3,3],ax[3,4]], n_mesh=100,
                         abs_max=2, levels=levels)
    plot_dressing_contour (extraB2, eta=1/2, M=extraB2(0)/10,
                        xDom=xDom_bottom, yDom=yDom_bottom,
                        axs=[ax[4,3],ax[4,4]], n_mesh=100,
                        abs_max=2, levels=levels)

    # Detail view
    xDom_top = (-0.04,1e-2,1e3)
    yDom_top = (-0.025,0.025)
    plot_dressing_contour (extraB1, eta=1/2, M=extraB1(0)/10,
                         xDom=xDom_top, yDom=yDom_top,
                         axs=[ax[0,3],ax[0,4]], n_mesh=100,
                         abs_max=2, levels=levels)
    plot_dressing_contour (extraB2, eta=1/2, M=extraB2(0)/10,
                        xDom=xDom_top, yDom=yDom_top,
                        axs=[ax[1,3],ax[1,4]], n_mesh=100,
                        abs_max=2, levels=levels)

    plot_dressing_contour (extraA1, eta=1/2, M=extraB1(0)/10,
                           xDom=xDom_top, yDom=yDom_top,
                           axs=[ax[0,0],ax[0,1]], n_mesh=100,
                           abs_max=2, levels=levels)
    plot_dressing_contour (extraA2, eta=1/2, M=extraB2(0)/10,
                          xDom=xDom_top, yDom=yDom_top,
                          axs=[ax[1,0],ax[1,1]], n_mesh=100,
                          abs_max=2, levels=levels)


    ax[0,1].yaxis.tick_right()
    ax[1,1].yaxis.tick_right()
    ax[0,4].yaxis.tick_right()
    ax[1,4].yaxis.tick_right()
    ax[3,1].yaxis.tick_right()
    ax[4,1].yaxis.tick_right()
    ax[3,4].yaxis.tick_right()
    ax[4,4].yaxis.tick_right()

    ax[0,1].yaxis.set_tick_params(rotation = 90)
    ax[1,1].yaxis.set_tick_params(rotation = 90)
    ax[0,4].yaxis.set_tick_params(rotation = 90)
    ax[1,4].yaxis.set_tick_params(rotation = 90)
    ax[3,1].yaxis.set_tick_params(rotation = 90)
    ax[4,1].yaxis.set_tick_params(rotation = 90)
    ax[3,4].yaxis.set_tick_params(rotation = 90)
    ax[4,4].yaxis.set_tick_params(rotation = 90)

    ax[0,0].yaxis.set_ticks([])
    ax[1,0].yaxis.set_ticks([])
    ax[0,1].yaxis.set_ticks([])
    ax[1,1].yaxis.set_ticks([])
    ax[0,0].xaxis.set_ticks([])
    ax[0,1].xaxis.set_ticks([])
    ax[3,0].yaxis.set_ticks([])
    ax[4,0].yaxis.set_ticks([])
    ax[3,1].yaxis.set_ticks([])
    ax[4,1].yaxis.set_ticks([])
    ax[3,0].xaxis.set_ticks([])
    ax[3,1].xaxis.set_ticks([])

    ax[0,3].yaxis.set_ticks([])
    ax[1,3].yaxis.set_ticks([])
    ax[0,3].xaxis.set_ticks([])
    ax[0,4].xaxis.set_ticks([])
    ax[3,3].yaxis.set_ticks([])
    ax[4,3].yaxis.set_ticks([])
    ax[3,3].xaxis.set_ticks([])
    ax[3,4].xaxis.set_ticks([])

    ax[0,4].yaxis.set_major_locator(plt.MaxNLocator(3))
    ax[1,4].yaxis.set_major_locator(plt.MaxNLocator(3))
    ax[1,0].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[1,3].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[3,4].yaxis.set_major_locator(plt.MaxNLocator(3))
    ax[4,4].yaxis.set_major_locator(plt.MaxNLocator(3))
    ax[4,0].xaxis.set_major_locator(plt.MaxNLocator(3))
    ax[4,3].xaxis.set_major_locator(plt.MaxNLocator(3))

    ax[0,0].set_title('$A(p^2)$')
    ax[0,3].set_title('$B(p^2)$')
    ax[0,0].set_ylabel('$m=0~GeV$')
    ax[1,0].set_ylabel('$m=0.115~GeV$')

    ax[4,0].set_xlabel('$Re~p^2~/~GeV$')
    ax[4,3].set_xlabel('$Re~p^2~/~GeV$')
    ax[3,0].set_ylabel('$m=0~GeV$')
    ax[4,0].set_ylabel('$m=0.115~GeV$')

    cbar = fig.colorbar(CS, ax=ax.ravel().tolist())
    cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(), rotation=90)
    cbar.ax.set_ylabel('absolute value')

    #plt.savefig(f'graphics/parabolaRegion.pdf', bbox_inches='tight')
    plt.show()

if check_Literature_numbers:
    params = [(0.4,45,0),(0.4,45,5e-3),(0.4,45,0.12),
              (0.45,25,0),(0.45,25,5e-3),(0.45,25,0.12),
              (0.5,16,0),(0.5,16,5e-3),(0.5,16,0.12)]

    for p in params:
        dressing = DressingFunctions(A_init,B_init,D=p[1],omega=p[0],m=p[2])
        dressing.solve_chebyshev(n=N, type='1st', res=1e-15)
        A, B = dressing.AB_fun_int(M=7)
        M0 = B(0)/A(0)

        print(f'{p}: {M0:.3f}')
