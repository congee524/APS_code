#!usr/bin/python
# this script is used for generating orders
#
# python OrderGenerator.py [name=filename] [number=number_of_orders] [importance=true/false]
#
# Irrelevant case and order, the default is name=data number=60 importance=true
#

import pandas as pd
import sys
import random
import datetime

def GenerateARecord(id, needImportance = True):
    modelNum = random.randint(1,10)
    quantity = random.randint(5,100)
    profit = quantity * 1.2 + (1+random.uniform(-0.1,0.1))
    profit = round(profit,2)
    ddlDelta = int(quantity * 1.5)
    now = datetime.datetime.now()
    delta = datetime.timedelta(days=ddlDelta)
    ddl = now + delta
    ddl = ddl.strftime('%Y-%m-%d')
    toc = random.randint(0,30)
    mvc = random.randint(100,3000)
    if needImportance == True:
        importance = int(toc/float(30)*2.5 + mvc/float(3000)*2.5)
        return (id,modelNum,quantity,profit,ddl,toc,mvc,importance)
    return (id,modelNum,quantity,profit,ddl,toc,mvc)

# parse the parameter
importance = True
fileName = 'data'
orderNumber = 60
errorsArised = False
for i in range(1,len(sys.argv)):
    para = sys.argv[i].lower()
    para = para.split('=')
    if para[0] == 'name':
        fileName = para[1]
    elif para[0] == 'number':
        orderNumber = int(para[1])
        while orderNumber <= 0:
            orderNumber = int(input("Please enter a positive number for order quantity : "))
    elif para[0] == 'importance':
        if para[1] == 'false':
            importance = False
    else:
        errorsArised = True
        print('parameter error! unknown parameter.')

# define the header
attributes = ['id','model_number','quantity','profit','deadline','times_of_cooperation','market_value_of_the_customer']
if importance == True:
    attributes.append('importance')

# generate orders
DataSet = []
for i in range(1,orderNumber+1):
    DataSet.append(GenerateARecord(i,importance))
dataFrame = pd.DataFrame(data = DataSet, columns = attributes)

# write the file
if errorsArised == True :
    print('using default parameters.')
print('order data generated. Named ' + fileName + ".csv")
dataFrame.to_csv(fileName + '.csv',index = False)
