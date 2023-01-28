import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#from common import BaselineDatabase, LogFileDatabase, geomean, draw_grouped_bar_chart, to_str_round, throughput_to_cost
#from shape_configs import shape_dict


if __name__ == "__main__":
    x = [0, 1, 2, 3, 4,5,6,7,8,9]
    y11 = [7545132.00000000,7571136.00000000,7604560.00000000,7696074.00000000,7690468.00000000,7609492.00000000,7514986.00000000,7572178.00000000,7487020.00000000,7659698.00000000]
    y12 = [7545132.00000000,7581428.00000000,7605512.00000000,7605922.00000000,7551770.00000000,7593464.00000000,7534246.00000000,7471666.00000000,7441800.00000000,7367222.00000000]
    
    y21=[4013276.00000000,4307864.00000000,4410668.00000000,4428130.00000000,4336560.00000000,4442604.00000000,4382420.00000000,4403188.00000000,4410414.00000000,4388752.00000000] 
    y31=[2093624.00000000,2086494.00000000,2057952.00000000,2052152.00000000,1993184.00000000,1917152.00000000,1823352.00000000,1752702.00000000,1695744.00000000,1634430.00000000]
    
    y41=[[4007500.00000000,4049332.00000000,4071170.00000000,4080024.00000000, 4073112.00000000, 4027580.00000000, 4037504.00000000, 4084474.00000000, 4037244.00000000, 4035488.00000000]
    ,[4210772.00000000,5015580.00000000, 5080884.00000000,5154416.00000000, 5087820.00000000, 5090836.00000000, 5069900.00000000, 4963394.00000000, 4940446.00000000, 4905932.00000000]
    ,[4047984.00000000,4140880.00000000, 4091128.00000000,3969552.00000000, 3972656.00000000, 4006052.00000000, 3961302.00000000, 3930202.00000000, 4034412.00000000, 4037396.00000000]
    ,[4095376.00000000,4245584.00000000, 4221596.00000000,4336916.00000000, 4264456.00000000, 4267628.00000000, 4204628.00000000, 4161568.00000000, 4086284.00000000, 3950084.00000000]
    ,[4087832.00000000,4223828.00000000, 4168964.00000000,4121268.00000000, 4136284.00000000, 4261080.00000000, 4195466.00000000, 4069442.00000000, 4128432.00000000, 4182664.00000000]
    ,[4148084.00000000,4258228.00000000, 4243744.00000000,4126478.00000000, 4229882.00000000, 4150612.00000000, 4204316.00000000, 4088332.00000000, 4029656.00000000, 4070444.00000000]
    ,[8262770.00000000,8492586.00000000, 8429604.00000000, 8473404.00000000, 8490168.00000000, 8503616.00000000, 8757856.00000000, 8580422.00000000, 8637314.00000000, 8633096.00000000]
    ,[4287948.00000000,4360068.00000000, 4344268.00000000, 4269182.00000000, 4165588.00000000, 4120772.00000000, 4151900.00000000, 4116176.00000000, 4043034.00000000, 3950014.00000000]
    ,[4167092.00000000,4225040.00000000, 4074532.00000000, 4107648.00000000, 4212102.00000000, 4212816.00000000, 4224396.00000000, 4375628.00000000, 4357924.00000000, 4435010.00000000]
    ,[4211720.00000000,4308212.00000000, 4264556.00000000, 4332204.00000000, 4327382.00000000, 4386844.00000000, 4333860.00000000, 4342894.00000000, 4336958.00000000, 4413594.00000000]
    ,[4052928.00000000,4101924.00000000, 4021688.00000000, 4097892.00000000, 4180736.00000000, 4072608.00000000, 4010192.00000000, 4105044.00000000, 3977088.00000000, 3950960.00000000]
    ,[8287560.00000000,8327370.00000000, 8433932.00000000, 8237394.00000000, 8284452.00000000, 8288612.00000000, 8159640.00000000, 8135512.00000000, 8161744.00000000, 8053096.00000000]
    ,[4247732.00000000,4249584.00000000, 4258964.00000000, 4260516.00000000, 4269960.00000000, 4215492.00000000, 4152050.00000000, 4061828.00000000, 4029104.00000000, 4037502.00000000]
    ,[4169444.00000000,4176030.00000000, 4120352.00000000, 4090148.00000000, 4088912.00000000, 4045526.00000000, 4079750.00000000, 4135812.00000000, 4068712.00000000, 4039676.00000000]
    ,[4283432.00000000,4196200.00000000, 4189676.00000000, 4111340.00000000, 4120484.00000000, 4076916.00000000, 4025492.00000000, 4063676.00000000, 4068092.00000000, 4105620.00000000]
    ,[4174034.00000000,4095870.00000000, 3993732.00000000, 4083582.00000000, 4059548.00000000, 3976308.00000000, 3966316.00000000, 3986804.00000000, 3946160.00000000, 3843520.00000000]
    ,[4615796.00000000,4979528.00000000, 4948186.00000000, 4878876.00000000, 4752228.00000000, 4759330.00000000, 4918528.00000000, 4949284.00000000, 4930572.00000000, 4890180.00000000]
    ,[4191114.00000000,4178740.00000000, 4154012.00000000, 4170148.00000000, 4222696.00000000, 4168304.00000000, 4192612.00000000, 4189834.00000000, 4183248.00000000, 4123618.00000000]
    ,[4331136.00000000,4341072.00000000, 4263348.00000000, 4261026.00000000, 4281448.00000000, 4241774.00000000, 4208912.00000000, 4197526.00000000, 4195296.00000000, 4259176.00000000]
    ,[4134384.00000000,4076426.00000000, 4077856.00000000, 4031804.00000000, 3979984.00000000, 3925638.00000000, 3946916.00000000, 3915916.00000000, 3947756.00000000, 3889436.00000000]
    ,[4342334.00000000,4159244.00000000, 4113300.00000000, 3906932.00000000, 3759492.00000000, 3803332.00000000, 3697916.00000000, 3576004.00000000, 3446588.00000000, 3404352.00000000]
    ,[4520308.00000000,4789404.00000000, 4740700.00000000, 4815060.00000000, 4799374.00000000, 4748672.00000000, 4705780.00000000, 4704780.00000000, 4668710.00000000, 4741852.00000000]
    ,[2408984.00000000,2387766.00000000, 2380214.00000000, 2350254.00000000, 2356284.00000000, 2377836.00000000, 2486936.00000000, 2505054.00000000, 2523990.00000000, 2491814.00000000]
    ,[2408318.00000000,2290508.00000000, 2296088.00000000, 2272206.00000000, 2302286.00000000, 2259942.00000000, 2337144.00000000, 2275712.00000000, 2239740.00000000, 2239062.00000000]
    ,[2298254.00000000,2292918.00000000, 2329470.00000000, 2436848.00000000, 2422284.00000000, 2506460.00000000, 2409984.00000000, 2357088.00000000, 2379608.00000000, 2351094.00000000]
    ,[2232068.00000000,2274726.00000000, 2203900.00000000, 2197206.00000000, 2278448.00000000, 2278748.00000000, 2109152.00000000, 2070926.00000000, 2091704.00000000, 2117742.00000000]]
    """
    [8134724.00000000,8313680.00000000, 8288468.00000000, 8341492.00000000, 8231564.00000000, 8177132.00000000, 8303222.00000000, 8128502.00000000, 8092920.00000000, 8204974.00000000]
    """
    

    y22=[4013276.00000000,4329152.00000000,4355716.00000000,4202420.00000000,4106972.00000000,4277712.00000000,4342528.00000000,4230236.00000000,4039968.00000000,4114064.00000000]
    y32=[2093624.00000000,2080224.00000000,2063352.00000000,2020320.00000000,1979054.00000000,1863224.00000000,1780344.00000000,1609080.00000000,1483320.00000000,1355390.00000000]
    
    y42=[[4007500.00000000, 4005792.00000000, 3953468.00000000, 4098218.00000000, 4067868.00000000, 4000790.00000000, 3903064.00000000, 3951122.00000000, 3984896.00000000, 3814664.00000000]
    ,[4210772.00000000, 4959794.00000000, 4889586.00000000, 4904892.00000000, 4937710.00000000, 4867512.00000000, 4762382.00000000, 4697450.00000000, 4658372.00000000, 4663744.00000000]
    ,[4047984.00000000, 4109708.00000000, 4119580.00000000, 4035432.00000000, 3902794.00000000, 3943760.00000000, 4007546.00000000, 4036778.00000000, 4004812.00000000, 3996800.00000000]
    ,[4095376.00000000, 4185060.00000000, 4061196.00000000, 4107508.00000000, 4038060.00000000, 4127742.00000000, 4012292.00000000, 4049142.00000000, 4099208.00000000, 4083524.00000000]
    ,[4087832.00000000, 4085048.00000000, 4000826.00000000, 3908432.00000000, 3876304.00000000, 3921490.00000000, 4044618.00000000, 4183760.00000000, 4248416.00000000, 4264558.00000000]
    ,[4148084.00000000, 4230840.00000000, 4188032.00000000, 4098714.00000000, 4120464.00000000, 4128566.00000000, 4175600.00000000, 4128398.00000000, 4071778.00000000, 4011840.00000000]
    ,[8262770.00000000, 8343986.00000000, 8291580.00000000, 8315946.00000000, 8158406.00000000, 8127270.00000000, 8089320.00000000, 7926752.00000000, 8178938.00000000, 8360210.00000000]
    ,[4287948.00000000, 4324756.00000000, 4389380.00000000, 4320068.00000000, 4247422.00000000, 4208528.00000000, 4281830.00000000, 4286260.00000000, 4281944.00000000, 4257770.00000000]
    ,[4167092.00000000, 4268878.00000000, 4269580.00000000, 4211036.00000000, 4258448.00000000, 4217950.00000000, 4238626.00000000, 4149968.00000000, 4124300.00000000, 4120368.00000000]
    ,[4211720.00000000, 4163376.00000000, 4162318.00000000, 4149852.00000000, 4170716.00000000, 4150210.00000000, 4051048.00000000, 3997964.00000000, 3960000.00000000, 3927704.00000000]
    ,[4052928.00000000, 4018496.00000000, 4007924.00000000, 4049972.00000000, 4091048.00000000, 4016652.00000000, 4053620.00000000, 4080440.00000000, 4061050.00000000, 4045148.00000000]
    ,[8287560.00000000, 8086992.00000000, 8128938.00000000, 7834472.00000000, 7755032.00000000, 7654544.00000000, 7748732.00000000, 7780522.00000000, 7698692.00000000, 7577186.00000000]
    ,[4247732.00000000, 4310356.00000000, 4270852.00000000, 4300236.00000000, 4129412.00000000, 4088052.00000000, 4049888.00000000, 4056710.00000000, 4122900.00000000, 4153524.00000000]
    ,[4169444.00000000, 4118024.00000000, 4068716.00000000, 4087162.00000000, 4123714.00000000, 4097536.00000000, 4141508.00000000, 4126596.00000000, 4130772.00000000, 4111064.00000000]
    ,[4283432.00000000, 4247264.00000000, 4243078.00000000, 4150970.00000000, 4044604.00000000, 3988958.00000000, 3982670.00000000, 4057412.00000000, 4068998.00000000, 3949862.00000000]
    ,[4174034.00000000, 4179332.00000000, 4221284.00000000, 4166356.00000000, 4215798.00000000, 4132588.00000000, 4058912.00000000, 4042338.00000000, 4131128.00000000, 4159012.00000000]
    ,[4615796.00000000, 4936444.00000000, 5005482.00000000, 4989416.00000000, 4986800.00000000, 5006120.00000000, 4888792.00000000, 4945284.00000000, 4761752.00000000, 4801942.00000000]
    ,[4191114.00000000, 4195964.00000000, 4097844.00000000, 4026556.00000000, 3978036.00000000, 3982570.00000000, 3990428.00000000, 3958820.00000000, 3997638.00000000, 4005584.00000000]
    ,[4331136.00000000, 4248020.00000000, 4199408.00000000, 4151422.00000000, 4087984.00000000, 4069000.00000000, 4075984.00000000, 4009184.00000000, 3935660.00000000, 4014404.00000000]
    ,[4134384.00000000, 4093612.00000000, 4063218.00000000, 4007458.00000000, 4019776.00000000, 4049606.00000000, 3975156.00000000, 3910952.00000000, 3926476.00000000, 3877610.00000000]
    ,[4342334.00000000, 4248506.00000000, 4110554.00000000, 3977508.00000000, 3889096.00000000, 3706230.00000000, 3492706.00000000, 3385284.00000000, 3328128.00000000, 3196460.00000000]
    ,[4520308.00000000, 4937268.00000000, 4913200.00000000, 4842820.00000000, 4859536.00000000, 4844596.00000000, 4854924.00000000, 4760684.00000000, 4785120.00000000, 4776332.00000000]
    ,[2408984.00000000, 2488958.00000000, 2482166.00000000, 2459744.00000000, 2477054.00000000, 2538844.00000000, 2520524.00000000, 2548798.00000000, 2507054.00000000, 2577438.00000000]
    ,[2408318.00000000, 2358534.00000000, 2396654.00000000, 2418062.00000000, 2370236.00000000, 2260006.00000000, 2254560.00000000, 2281644.00000000, 2343580.00000000, 2348088.00000000]
    ,[2298254.00000000, 2327036.00000000, 2271638.00000000, 2267550.00000000, 2279950.00000000, 2218878.00000000, 2257902.00000000, 2337254.00000000, 2361366.00000000, 2256224.00000000]
    ,[2232068.00000000, 2264974.00000000, 2234108.00000000, 2369078.00000000, 2247614.00000000, 2129072.00000000, 2159672.00000000, 2220198.00000000, 2214764.00000000, 2219952.00000000]]
    """ 
    [4505388.00000000, 4905276.00000000, 4938088.00000000, 4839334.00000000, 4861168.00000000, 4759986.00000000, 4820972.00000000, 4804516.00000000, 4810088.00000000, 4717536.00000000]
    """

    ya1=[]
    ya2=[]
    for i in range(0,10):
        tmp=0.00000000
        for j in range(0,26):
            tmp+=y41[i][j];
        ya1.append(tmp)

    for i in range(0,10):
        tmp=0.00000000
        for j in range(0,26):
            tmp+=y42[i][j];
        ya1.append(tmp)

    print(ya1,ya2)

#     name_list=['Ansor','Ansor-DPC']
#     fig, ax = plt.subplots() 

#     gs = gridspec.GridSpec(3, 1)
#     ax1=plt.subplot(gs[0])
#     ax1.plot(x, y11,
#             color = 'green',
#             linewidth = 3)
#     ax1.plot(x, y12,
#         color = 'red',
#         linewidth = 3)

#     ax2=plt.subplot(gs[1])
#     ax2.plot(x, y21,
#             color = 'green',
#             linewidth = 3)
#     ax2.plot(x, y22,
#         color = 'red',
#         linewidth = 3)

#     ax3=plt.subplot(gs[2])
#     ax3.plot(x, y31,
#             color = 'green',
#             linewidth = 3)
#     ax3.plot(x, y32,
#         color = 'red',
#         linewidth = 3)


    ax1.set_ylabel("Diversity Population ", fontsize=18)
    ax1.text(0.5, -0.12, 'Iterations', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=ax.yaxis.label.get_size())
    ax1.legend(name_list,
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