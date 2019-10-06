# -*- coding:utf-8 -*-

import random
import math
from operator import itemgetter
import numpy as np
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
            for i in range(0, self.totnum_task):
                while True:
                    ran_line_id = random.randint(0, self.line_num - 1)
                    if self.yield_time[ran_line_id][self.task_ind[i]] == -1:
                        continue
                    geneinfo[ran_line_id].append(i)
                    break

            fitness = self.evaluate(geneinfo)  # evaluate each chromosome
            # store the chromosome and its fitness
            pop.append({'data': geneinfo, 'fitness': fitness})

        self.pop = pop
        # store the best chromosome in the population
        self.bestindividual = self.selectBest(self.pop)
        print("  Finish initial ")

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
        line_req = -1
        prod_req = -1
        if pre_pos[1] > 0:
            line_req = individual[pre_pos[0]][pre_pos[1] - 1]
        if task_id != np.where(self.prod_ind == self.prod_ind[task_id])[0][0]:
            prod_req = task_id - 1

        finish_time_status[task_id] = -1
        line_req_time = -1
        prod_req_time = -1
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
            if ((pre_pos[0] == last_pos[0]) and (pre_pos[1] == last_pos[1] + 1)):
                isn_proce = False

        if line_req_time == -1 and prod_req_time == -1:
            isn_proce = False

        finish_time_status[task_id] = 0
        pre_task_time = self.yield_time[pre_pos[0]][self.task_ind[task_id]]
        if ((prod_req == -1) and (line_req == -1)):
            finish_time_task[task_id] = pre_task_time
        else:
            finish_time_task[task_id] = max(
                prod_req_time, line_req_time) + isn_proce*self.ex_time + pre_task_time
        return finish_time_task[task_id]

    def evaluate(self, individual):
        # return fitness value
        # individual is the task id on each line
        # print("  evaluate")
        # print("geninfo: %s" % individual)
        individual = np.array(individual)
        line_preworktime = [[] for _ in range(self.prod_num)]
        for i in range(self.prod_num):
            line_preworktime[i] = [self.yield_time[i][j]
                                   for j in self.task_ind[individual[i]]]
        line_worktime = np.array([sum(i) for i in line_preworktime])

        finish_time_task = [-1 for _ in range(self.totnum_task)]
        first_task_ind = []
        last_task_ind = []
        for i in range(self.prod_num):
            first_task_ind.append(np.where(self.prod_ind == i)[0][0])
            last_task_ind.append(np.where(self.prod_ind == i)[0][-1])
        first_task_ind = np.array(first_task_ind)
        last_task_ind = np.array(last_task_ind)
        task_line_loca = self.get_pos(individual)

        finish_time_task = [-1 for _ in range(self.totnum_task)]
        finish_time_status = [0 for _ in range(self.totnum_task)]

        for i in range(self.totnum_task):
            if (self.get_finish_time(individual, i, task_line_loca, finish_time_task, finish_time_status) == -1):
                print("go wrong!")
                exit()
                return -1

        var_line = 0
        if (np.max(line_worktime) == np.min(line_worktime)):
            var_line = 0
        else:
            line_worktime = (line_worktime - np.min(line_worktime)) / \
                (np.max(line_worktime) - np.min(line_worktime))
            var_line = np.var(line_worktime)

        finish_time_task = np.array(finish_time_task)
        finish_time_prod = finish_time_task[last_task_ind]

        fitness = np.sum(finish_time_prod * self.prod_prio) * (1 + var_line**2)
        return fitness

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
        s_inds = sorted(individuals, key=itemgetter(
            "fitness"), reverse=True)  # sort the pop by the reference of 1/fitness
        # sum up the 1/fitness of the whole pop
        sum_fits = sum(1.0 / ind['fitness'] for ind in individuals)

        chosen = []
        for _ in range(k):
            # randomly produce a num in the range of [0, sum_fits]
            u = random.random() * sum_fits
            sum_ = 0
            for ind in s_inds:
                sum_ += 1.0 / ind['fitness']  # sum up the 1/fitness
                if sum_ > u:
                    # when the sum of 1/fitness is bigger than u, choose the one, which means u is in the range of [sum(1,2,...,n-1),sum(1,2,...,n)] and is time to choose the one ,namely n-th individual in the pop
                    chosen.append(ind)
                    break

        return chosen

    def disrupt_line(self, offspring):
        # print("  disrupt_line")
        geninfo = offspring['data']
        tarind_line = random.randint(0, self.line_num - 1)
        for _ in range(self.line_num * 2):
            if (len(geninfo[tarind_line]) >= 2):
                break
            tarind_line = random.randint(0, self.line_num - 1)
        if (len(geninfo[tarind_line]) < 2):
            return geninfo

        inline = geninfo[tarind_line].copy()
        tmp_line = inline.copy()
        left = -1
        right = -1
        # print("before geninfo[tarind_line]: %s" % geninfo[tarind_line])
        for _ in range(len(tmp_line) ** 2):
            random.shuffle(tmp_line)
            if (tmp_line[0] < tmp_line[1]):
                left = tmp_line[0]
                right = tmp_line[1]
            else:
                right = tmp_line[0]
                left = tmp_line[1]
            if self.prod_ind[left] == self.prod_ind[right]:
                continue
            # print("line %i left, right: (%i, %i)" % (tarind_line, left, right))
            # print("prod_ind %s" % self.prod_ind)
            # print("line %i (left, right): prod_ind (%i, %i)" %
            #       (tarind_line, self.prod_ind[left], self.prod_ind[right]))
            task_line_loca = self.get_pos(geninfo)
            right_ex_pos = task_line_loca[left]
            right_firsttask = np.where(
                self.prod_ind == self.prod_ind[right])[0][0]

            flag = 0
            for pre_task_ind in range(right_firsttask, right + 1):
                pre_pos = task_line_loca[pre_task_ind]
                for solve_task_ind in inline[right_ex_pos[1]:right]:
                    solve_prod_ind = self.prod_ind[solve_task_ind]
                    solve_firsttask = np.where(
                        self.prod_ind == solve_prod_ind)[0][0]
                    for pre_solve_task in range(solve_firsttask, solve_task_ind + 1):
                        pre_solve_pos = task_line_loca[pre_solve_task]
                        if ((pre_solve_pos[0] == pre_pos[0]) and (pre_pos[1] > pre_solve_pos[1])):
                            flag = 1
                            break
                    if (flag == 1):
                        break
                if (flag == 1):
                    break
            if flag == 1:
                continue

            # flag = 0
            # for pre_task_ind in range(left_firsttask, left):
            #     pre_pos = task_line_loca[pre_task_ind]
            #     if (pre_pos[0] == left_ex_pos[0]) and (pre_pos[1] > left_ex_pos[1]):
            #         flag = 1
            #         break
            # if flag == 1:
            #     continue

            del geninfo[right_ex_pos[0]][task_line_loca[right][1]]
            geninfo[right_ex_pos[0]].insert(right_ex_pos[1], right)

            break

        # print("after geninfo[tarind_line]: %s" % geninfo[tarind_line])
        return geninfo

    def disrupt_prod(self, offspring):
        geninfo = offspring['data']
        # print("before disrupt_prod geninfo: %s" % geninfo)
        tmp_line = list(range(self.line_num))
        if (len(tmp_line) < 2):
            return geninfo

        while True:
            random.shuffle(tmp_line)
            if (len(geninfo[tmp_line[0]]) > 0):
                break

        tmp_geninfo = copy.deepcopy(geninfo)

        inline = copy.deepcopy(geninfo[tmp_line[0]])
        exline = copy.deepcopy(geninfo[tmp_line[1]])

        tar, tar_task = 0, 0
        for i in range(len(inline) * 2):
            tar = random.randint(0, len(inline) - 1)
            tar_task = inline[tar]
            if (yield_time[tmp_line[1]][self.task_ind[tar_task]] != -1):
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
                        if (self.get_finish_time(tmp_geninfo, i, task_line_loca, finish_time_task, finish_time_status) == -1):
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

        # tar_firsttask = np.where(
        #     self.prod_ind == self.prod_ind[tar_task])[0][0]
        # tar_lasttask = np.where(
        #     self.prod_ind == self.prod_ind[tar_task])[0][-1]

        # task_line_loca = self.get_pos(geninfo)

        # bound_left, bound_right = 0, len(exline)
        # for pre_task_ind in range(tar_firsttask, tar_lasttask + 1):
        #     pre_pos = task_line_loca[pre_task_ind]
        #     if (pre_pos[0] == tmp_line[1]):
        #         if (pre_task_ind < tar_task):
        #             bound_left = pre_pos[1] + 1
        #         elif (pre_task_ind > tar_task):
        #             bound_right = pre_pos[1]
        #             break

        # geninfo[tmp_line[1]].insert(random.randint(
        #     bound_left, bound_right), tar_task)
        # del geninfo[tmp_line[0]][tar]

        # return geninfo

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

                print("first %s" % offspring['data'])
                # cross two individuals with probability CXPB
                if random.random() < self.para['LINEPB']:
                    disline_data = self.disrupt_line(offspring)
                    offspring['data'] = copy.deepcopy(disline_data)
                    print("second %s" % offspring['data'])

                # mutate an individual with probability MUTPB
                if random.random() < self.para['PRODPB']:
                    disprod_data = self.disrupt_prod(offspring)
                    offspring['data'] = copy.deepcopy(disprod_data)
                    print("third %s" % offspring['data'])

                offspring['fitness'] = self.evaluate(offspring['data'])
                nextoff.append(offspring)

            # The population is entirely replaced by the offspring
            self.pop = nextoff

            # Gather all the fitnesses in one list and print the stats
            fits = [ind['fitness'] for ind in self.pop]

            length = len(self.pop)
            mean = sum(fits) / length
            self.rec_mean.append(mean)
            sum2 = sum(x*x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5
            self.rec_std.append(std)
            best_ind = self.selectBest(self.pop)

            if best_ind['fitness'] < self.bestindividual['fitness']:
                self.bestindividual = best_ind

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
    yield_time = [[9, 8, -1, 7, 6], [32, 4, 8, 1, 4],
                  [1, -1, -1, -1, -1], [6, 6, -1, 4, -1]]
    prod_ind = [0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    task_ind = [4, 1, 0, 2, 0, 3, 3, 3, 1, 1, 0, 2, 4]
    prod_prio = [5, 3, 3]
    """
    parameter = {'LINEPB': 0.7, 'PRODPB': 0.7, 'NGEN': 50, 'popsize': 100, 'ex_time': 3, 'yield_time': yield_time, 'prod_ind': prod_ind,
                 'task_ind': task_ind, "prod_prio": prod_prio}
                 """
    parameter = {'LINEPB': 0.7, 'PRODPB': 0.7, 'NGEN': 50, 'popsize': 500, 'ex_time': 3, 'yield_time': yield_time, 'prod_ind': prod_ind,
                 'task_ind': task_ind, "prod_prio": prod_prio}
    run = GA(parameter)
    run.GA_main()
