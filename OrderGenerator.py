#!usr/bin/python
# this script is used for generating orders

import pandas as pd
import sys
import random
import datetime


def GenerateARecord():
    modelNum = random.randint(1, 10)
    quantity = random.randint(5, 100)
    profit = quantity * 1.2 + (1+random.uniform(-0.1, 0.1))
    profit = round(profit, 2)
    ddlDelta = int(quantity * 1.5)
    now = datetime.datetime.now()
    delta = datetime.timedelta(days=ddlDelta)
    ddl = now + delta
    ddl = ddl.strftime('%Y-%m-%d')
    toc = random.randint(0, 30)
    mvc = random.randint(100, 3000)
    importance = int(toc/float(30)*2.5 + mvc/float(3000)*2.5)
    return (modelNum, quantity, profit, ddl, toc, mvc, importance)


# define the header
attributes = ['model_num', 'quantity', 'profit',
              'deadline', 'cooperation', 'market_value', 'importance']
orderNumber = 60
if len(sys.argv) >= 2:
    orderNumber = int(sys.argv[1])
while orderNumber <= 0:
    orderNumber = int(
        input("Please enter a positive number for order quantity : "))

# generate orders
DataSet = []
for i in range(1, orderNumber+1):
    DataSet.append(GenerateARecord())
dataFrame = pd.DataFrame(data=DataSet, columns=attributes)

# write the file
print('order data generated. named data.csv')
dataFrame.to_csv('data.csv')
