# -*- coding:utf-8 -*-

import random
from operator import itemgetter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


class GA:
    '''
    This is a class of GA algorithm. 
    '''

    def __init__(self, parameter):
        '''
        Initialize the pop of GA algorithom and evaluate the pop by computing its' fitness value .
        The data structure of pop is composed of several individuals which has the form like that:

        {'Gene':a object of class Gene, 'fitness': 1.02(for example)}
        Representation of Gene is a list: [b s0 u0 sita0 s1 u1 sita1 s2 u2 sita2]

        '''
        self.para = parameter
        self.popsize = self.para['popsize']
        self.yield_time = np.array(self.para['yield_time'])
        self.prod_ind = np.array(self.para['prod_ind'])
        self.task_ind = np.array(self.para['task_ind'])
        self.prod_prio = np.array(self.para['prod_prio'])
        self.line_num = self.yield_time.shape[0]
        self.prod_num = self.prod_prio.shape[0]
        self.ex_time = self.para['ex_time']
        self.totnum_task = self.prod_ind.shape[0]
        self.rec_std = []
        self.rec_mean = []

        pop = []
        for _ in range(self.popsize):
            geneinfo = [[] for _ in range(self.line_num)]
            pre_task_ind = []
            last_task_ind = []

            for i in range(self.prod_num):
                pre_task_ind.append(np.where(self.prod_ind == i)[0][0])
                last_task_ind.append(np.where(self.prod_ind == i)[0][-1])
            pre_task_ind = np.array(pre_task_ind)
            last_task_ind = np.array(last_task_ind)

            finish_time_task = [-1 for _ in range(self.totnum_task)]
            task_line_loca = self.get_pos(geneinfo)
            while sum(pre_task_ind > last_task_ind) < self.prod_num:
                rand_prod_ind = random.randint(0, self.prod_num - 1)
                if pre_task_ind[rand_prod_ind] > last_task_ind[rand_prod_ind]:
                    continue
                while True:
                    rand_line_id = random.randint(0, self.line_num - 1)
                    if self.yield_time[rand_line_id][self.task_ind[pre_task_ind[rand_prod_ind]]] != -1:
                        geneinfo[rand_line_id].append(pre_task_ind[rand_prod_ind])
                        pre_task_ind[rand_prod_ind] += 1
                        break
            fitness = self.evaluate(geneinfo)
            pop.append({'data': geneinfo, 'fitness': fitness})
        self.pop = pop
        # store the best chromosome in the population
        self.bestindividual = self.selectBest(self.pop)
        print("  Finish initial ")


        """
        for i in range(0, self.totnum_task):
            while True:
                ran_line_id = random.randint(0, self.line_num - 1)
                if self.yield_time[ran_line_id][self.task_ind[i]] == -1:
                    continue
                geneinfo[ran_line_id].append(i)
                break
                # print("first %s" % offspring['data'])
                # cross two individuals with probability CXPB
        offspring = {'data': geneinfo, 'fitness': 0}
        k = random.randint(1, self.line_num)
        for _ in range(k):
            if random.random() < self.para['PRODPB']:
                disprod_data = self.disrupt_prod(offspring)
                offspring['data'] = disprod_data
                # print("third %s" % offspring['data'])

            if random.random() < self.para['LINEPB']:
                disline_data = self.disrupt_line(offspring)
                offspring['data'] = disline_data
                # print("second %s" % offspring['data'])
        fitness = self.evaluate(geneinfo)  # evaluate each chromosome
        offspring['fitness'] = fitness
        # store the chromosome and its fitness
        pop.append(offspring)
        """

    def selectBest(self, pop):
        '''
        select the best individual from pop
        '''
        s_inds = sorted(pop, key=itemgetter("fitness"), reverse=False)
        return s_inds[0]

    def selection(self, individuals, k):
        '''
        select two individuals from pop
        '''
        # print(individuals)
        fit_rank = np.array(pd.Series(ind['fitness'] for ind in individuals).rank(ascending=False))
        max_fit = np.max(fit_rank)
        fit_rank[fit_rank < max_fit / 4] = 1
        fit_rank[(fit_rank < max_fit / 2) & (fit_rank >= max_fit / 4)] = 2
        fit_rank[(fit_rank < 3 * max_fit / 4) & (fit_rank >= max_fit / 2)] = 3
        fit_rank[fit_rank >= 3 * max_fit / 4] = 4
        max_fit = np.max(fit_rank)
        chosen = []
        for _ in range(k):
            while True:
                # randomly select an individual with uniform probability
                ind = int(self.popsize * random.random())
                # with probability wi/wmax to accept the selection
                if random.random() <= fit_rank[ind] / max_fit:
                    chosen.append(individuals[ind])
                    break
        return chosen

    def get_pos(self, individual):
        task_line_loca = [-1 for _ in range(self.totnum_task)]
        for ind1, li in enumerate(individual):
            for ind2, pos in enumerate(li):
                task_line_loca[pos] = (ind1, ind2)
        return task_line_loca

    def get_finish_time(self, individual, task_id, task_line_loca, finish_time_task, finish_time_status):
        if finish_time_task[task_id] != -1:
            return finish_time_task[task_id]
        if finish_time_status[task_id] == -1:
            return - 1

        pre_pos = task_line_loca[task_id]
        line_req, prod_req, line_req_time, prod_req_time = -1, -1, -1, -1
        if pre_pos[1] > 0:
            line_req = individual[pre_pos[0]][pre_pos[1] - 1]
        if task_id != np.where(self.prod_ind == self.prod_ind[task_id])[0][0]:
            prod_req = task_id - 1

        finish_time_status[task_id] = -1

        if line_req != -1:
            line_req_time = self.get_finish_time(
                individual, line_req, task_line_loca, finish_time_task, finish_time_status)
            if line_req_time == -1:
                return - 1
        if prod_req != -1:
            prod_req_time = self.get_finish_time(
                individual, prod_req, task_line_loca, finish_time_task, finish_time_status)
            if prod_req_time == -1:
                return - 1

        isn_proce = True
        if prod_req != -1:
            last_pos = task_line_loca[prod_req]
            if (pre_pos[0] == last_pos[0]) and (pre_pos[1] == last_pos[1] + 1):
                isn_proce = False
        if line_req_time == -1 and prod_req_time == -1:
            isn_proce = False

        finish_time_status[task_id] = 0
        pre_task_time = self.yield_time[pre_pos[0]][self.task_ind[task_id]]
        if (prod_req == -1) and (line_req == -1):
            finish_time_task[task_id] = pre_task_time
        else:
            finish_time_task[task_id] = max(
                prod_req_time, line_req_time) + isn_proce * self.ex_time + pre_task_time

        return finish_time_task[task_id]

    def evaluate(self, individual):
        # return fitness value
        # individual is the task id on each line
        individual = np.array(individual)
        line_preworktime = [[] for _ in range(self.line_num)]
        for i in range(self.line_num):
            line_preworktime[i] = [self.yield_time[i][j]
                                   for j in self.task_ind[individual[i]]]
        line_worktime = np.array([sum(i) for i in line_preworktime])

        last_task_ind = []
        for i in range(self.prod_num):
            last_task_ind.append(np.where(self.prod_ind == i)[0][-1])
        last_task_ind = np.array(last_task_ind)
        task_line_loca = self.get_pos(individual)

        finish_time_task = [-1 for _ in range(self.totnum_task)]
        finish_time_status = [0 for _ in range(self.totnum_task)]

        for i in range(self.totnum_task):
            if self.get_finish_time(individual, i, task_line_loca, finish_time_task, finish_time_status) == -1:
                print("Populations with unsatisfactory constraints emerged!")
                exit()

        if np.max(line_worktime) == np.min(line_worktime):
            var_line = 0
        else:
            line_worktime = (line_worktime - np.min(line_worktime)) / \
                            (np.max(line_worktime) - np.min(line_worktime))
            var_line = np.var(line_worktime)

        finish_time_task = np.array(finish_time_task)
        finish_time_prod = finish_time_task[last_task_ind]

        fitness = np.sum(finish_time_prod * self.prod_prio) * (1 + var_line ** 2)
        return fitness

    def disrupt_line(self, offspring):
        # print("  disrupt_line")
        geninfo = offspring['data']
        tarind_line = random.randint(0, self.line_num - 1)
        for _ in range(self.line_num * 2):
            if len(geninfo[tarind_line]) >= 2:
                break
            tarind_line = random.randint(0, self.line_num - 1)
        if len(geninfo[tarind_line]) < 2:
            return geninfo

        inline = geninfo[tarind_line].copy()
        ori_task_line_loca = self.get_pos(geninfo)

        # print("before geninfo[tarind_line]: %s" % geninfo[tarind_line])
        for _ in range(len(inline) ** 2):
            random.shuffle(inline)
            left = inline[0]
            right = inline[1]
            geninfo[tarind_line][ori_task_line_loca[left][1]] = right
            geninfo[tarind_line][ori_task_line_loca[right][1]] = left

            finish_time_task = [-1 for _ in range(self.totnum_task)]
            finish_time_status = [0 for _ in range(self.totnum_task)]
            task_line_loca = self.get_pos(geninfo)

            flag = False
            for i in range(self.totnum_task):
                if self.get_finish_time(geninfo, i, task_line_loca, finish_time_task, finish_time_status) == -1:
                    flag = True
                    break

            if flag:
                geninfo[tarind_line][ori_task_line_loca[left][1]] = left
                geninfo[tarind_line][ori_task_line_loca[right][1]] = right
                continue
            else:
                return geninfo
        return geninfo

    def disrupt_prod(self, offspring):
        geninfo = offspring['data']
        # print("before disrupt_prod geninfo: %s" % geninfo)
        tmp_line = list(range(self.line_num))
        if len(tmp_line) < 2:
            return geninfo

        while True:
            random.shuffle(tmp_line)
            if len(geninfo[tmp_line[0]]) > 0:
                break

        inline = copy.deepcopy(geninfo[tmp_line[0]])
        exline = copy.deepcopy(geninfo[tmp_line[1]])

        for i in range(len(inline) * 2):
            tar = random.randint(0, len(inline) - 1)
            tar_task = inline[tar]
            if yield_time[tmp_line[1]][self.task_ind[tar_task]] != -1:
                bound_left, bound_right = 0, len(exline)
                rand_time = (bound_right - bound_left + 1) * 2
                for _ in range(rand_time):
                    # must use deepcopy in that del will influence the original list with shallow copy
                    tmp_geninfo = copy.deepcopy(geninfo)
                    tmp_geninfo[tmp_line[1]].insert(random.randint(
                        bound_left, bound_right), tar_task)
                    del tmp_geninfo[tmp_line[0]][tar]

                    finish_time_task = [-1 for _ in range(self.totnum_task)]
                    finish_time_status = [0 for _ in range(self.totnum_task)]
                    task_line_loca = self.get_pos(tmp_geninfo)

                    flag = False
                    for i in range(self.totnum_task):
                        if (self.get_finish_time(tmp_geninfo, i, task_line_loca, finish_time_task,
                                                 finish_time_status) == -1):
                            flag = True
                            break
                    if flag:
                        continue
                    else:
                        geninfo = copy.deepcopy(tmp_geninfo)
                        # print("after disrupt_prod geninfo: %s" % geninfo)
                        # print()
                        return geninfo

        return geninfo

    def GA_main(self):
        '''
        main frame work of GA
        '''

        popsize = self.para['popsize']

        print("Start of evolution")

        # Begin the evolution
        for g in range(self.para['NGEN']):

            print("-- Generation %i --" % g)

            # Apply selection based on their converted fitness
            selectpop = self.selection(self.pop, popsize)

            nextoff = []
            for _ in range(self.popsize):
                # Apply crossover and mutation on the offspring

                # Select one individuals
                offspring = random.choice(selectpop)

                # print("first %s" % offspring['data'])
                # cross two individuals with probability CXPB
                # k = random.randint(1, self.line_num * 2)
                k = 1
                for _ in range(k):
                    if random.random() < self.para['PRODPB']:
                        disprod_data = self.disrupt_prod(offspring)
                        offspring['data'] = disprod_data
                        # print("third %s" % offspring['data'])

                    if random.random() < self.para['LINEPB']:
                        disline_data = self.disrupt_line(offspring)
                        offspring['data'] = disline_data
                        # print("second %s" % offspring['data'])

                # mutate an individual with probability MUTPB

                offspring['fitness'] = self.evaluate(offspring['data'])
                nextoff.append(offspring)

            # The population is entirely replaced by the offspring
            self.pop = nextoff

            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]

            length = len(self.pop)
            mean = sum(fits) / length
            self.rec_mean.append(mean)
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            self.rec_std.append(std)
            best_ind = self.selectBest(self.pop)

            if best_ind['fitness'] < self.bestindividual['fitness']:
                self.bestindividual = copy.deepcopy(best_ind)

            print("Best individual found is %s, %s" % (
                self.bestindividual['data'], self.bestindividual['fitness']))
            print("  Min fitness of current pop: %s" % min(fits))
            print("  Max fitness of current pop: %s" % max(fits))
            print("  Avg fitness of current pop: %s" % mean)
            print("  Std of currrent pop: %s" % std)

        print("-- End of (successful) evolution --")
        x = list(range(1, self.para['NGEN'] + 1))
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, self.rec_mean, 'r', label='mean value')
        ax1.legend(loc=1)
        ax1.set_ylabel('mean value')
        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(x, self.rec_std, 'b', label="standard deviation")
        ax2.legend(loc=2)
        ax2.set_ylabel('standard deviation')
        ax2.set_xlabel('Generation')
        plt.title('Trend of population standard deviation and mean value')
        plt.show()


if __name__ == "__main__":
    # control parameters
    # prod_time = [[] for _ in range(parameter['prod_line_num'])]
    yield_time = [[9, 8, -1, 7, 6, 3, 2, 5], [32, 4, 8, 1, 4, 6, 3, 4],
                  [1, -1, -1, -1, -1, 2, 9, -1], [6, 6, -1, 4, -1, -1, 4, 5], [3, 6, 1, 6, -1, 9, -1, 10]]
    prod_ind = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5]
    task_ind = [4, 1, 0, 2, 0, 5, 3, 3, 6, 1, 0, 2, 4, 7, 7, 5, 1, 4, 3, 2, 0, 6, 2]
    prod_prio = [5, 3, 4, 1, 2, 2]
    """
    parameter = {'LINEPB': 0.7, 'PRODPB': 0.7, 'NGEN': 50, 'popsize': 100, 'ex_time': 3, 'yield_time': yield_time, 'prod_ind': prod_ind,
                 'task_ind': task_ind, "prod_prio": prod_prio}
                 """
    parameter = {'LINEPB': 0.7, 'PRODPB': 0, 'NGEN': 50, 'popsize': 200, 'ex_time': 3, 'yield_time': yield_time,
                 'prod_ind': prod_ind,
                 'task_ind': task_ind, "prod_prio": prod_prio}
    run = GA(parameter)
    run.GA_main()
