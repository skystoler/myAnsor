import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

#from common import BaselineDatabase, LogFileDatabase, geomean, draw_grouped_bar_chart, to_str_round, throughput_to_cost
#from shape_configs import shape_dict


if __name__ == "__main__":
    x1 = [0, 1, 2, 3, 4,5,6,7,8,9]
    y1 = [7545132.00000000 ,7571136.00000000 ,7604560.00000000 ,7696074.00000000 ,7690468.00000000 ,7609492.00000000 ,7514986.00000000 ,7572178.00000000 ,7487020.00000000 ,7659698.00000000 ]
    x2 = [0, 1, 2, 3, 4,5,6,7,8,9]
    y2 = [7545132.00000000 ,7581428.00000000 ,7605512.00000000 ,7605922.00000000 ,7551770.00000000 ,7593464.00000000 ,7534246.00000000 ,7471666.00000000 ,7441800.00000000 ,7367222.00000000 ]
    name_list=['Ansor','Ansor-DPC']
    fig, ax = plt.subplots() 
    plt.plot(x1, y1,
            color = 'red',
            linewidth = 3)
    plt.plot(x2, y2,
        color = 'black',
        linewidth = 3)
    ax.set_ylim(730000.00000000 ,800000.00000000) 
    ax.set_ylabel("Population Diversity", fontsize=18)
    ax.text(0.5, -0.12, 'Iterations', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=ax.yaxis.label.get_size())
    ax.legend(name_list,
            fontsize=18,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.17),
            ncol=3,
            handlelength=1.0,
            handletextpad=0.5,
            columnspacing=1.1)
    fig.set_size_inches((11, 5))
    fig.savefig("population-diversity.png", bbox_inches='tight')
    plt.show()