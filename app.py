from flask import Flask, render_template,request,make_response
import plotly
import plotly.graph_objs as go
import mysql.connector
from mysql.connector import Error
import sys
# Standalone simple linear regression example
from math import sqrt
import pandas as pd
import numpy as np
import json  #json request
from werkzeug.utils import secure_filename
import os
import csv #reading csv
import geocoder
from random import randint
from metrics import Processor

mon1=1.5
mon6=0.8
mon12=8

app = Flask(__name__)


@app.route('/')
def index():
    os.system("python Kmeans.py")
    return render_template('dataloader.html')

@app.route('/index')
def indexnew():    
    return render_template('index.html')






    





@app.route('/dataloader')
def dataloader():
    return render_template('dataloader.html')



@app.route('/cleardataset', methods = ['POST'])
def cleardataset():
    connection = mysql.connector.connect(host='localhost',database='rainfalldb',user='root',password='')
    cursor = connection.cursor()
    query="delete from rainfalldb"
    cursor.execute(query)
    connection.commit()      
    connection.close()
    cursor.close()
    return render_template('dataloader.html')



@app.route('/uploadajax', methods =  ['GET','POST'])
def upldfile():
    print("request :"+str(request), flush=True)
    if request.method == 'POST':
        connection = mysql.connector.connect(host='localhost',database='rainfalldb',user='root',password='')
        cursor = connection.cursor()
    
        prod_mas = request.files['prod_mas']
        filename = secure_filename(prod_mas.filename)
        prod_mas.save(os.path.join("E:\\Upload\\", filename))

        #csv reader
        fn = os.path.join("E:\\Upload\\", filename)

        # initializing the titles and rows list 
        fields = [] 
        rows = []
        
        with open(fn, 'r') as csvfile:
            # creating a csv reader object 
            csvreader = csv.reader(csvfile)  
  
            # extracting each data row one by one 
            for row in csvreader:
                rows.append(row)
                print(row)

        try:     
            #print(rows[1][1])       
            for row in rows[1:]: 
                # parsing each column of a row
                if row[0][0]!="":                
                    query="";
                    query="insert into rainfalldata values (";
                    for col in row: 
                        query =query+"'"+col+"',"
                    query =query[:-1]
                    query=query+");"
                print("query :"+str(query), flush=True)
                cursor.execute(query)
                connection.commit()
        except:
            print("An exception occurred")
        csvfile.close()
        
        print("Filename :"+str(prod_mas), flush=True)       
        
        
     

        #Fetch the rainfeed
        

        sq_query="SELECT Year,(SUM(JAN)+SUM(FEB)+SUM(MAR)+SUM(APR)+SUM(MAY)+SUM(JUN)+SUM(JUL)+SUM(AUG)+SUM(SEP)+SUM(OCT)+SUM(NOV)+SUM(DECEM)) AS Total FROM rainfalldata GROUP BY Year"
        cursor.execute(sq_query)
        raindata = cursor.fetchall()
        connection.close()
        cursor.close()


        
        return render_template('dataloader.html',data="Data loaded successfully",raindata=raindata)
       


@app.route('/procdataset', methods =  ['GET','POST'])
def procdataset():
    print("request :"+str(request), flush=True)
    if request.method == 'GET':
        connection = mysql.connector.connect(host='localhost',database='rainfalldb',user='root',password='')
        cursor = connection.cursor()
    
        #Fetch the rainfeed
        

        sq_query="SELECT Year,(SUM(JAN)+SUM(FEB)+SUM(MAR)+SUM(APR)+SUM(MAY)+SUM(JUN)+SUM(JUL)+SUM(AUG)+SUM(SEP)+SUM(OCT)+SUM(NOV)+SUM(DECEM)) AS Total FROM rainfalldata GROUP BY Year"
        cursor.execute(sq_query)
        print(sq_query)
        raindata = cursor.fetchall()


        sq_query="SELECT Distinct SUBDIVISION from rainfalldata"
        cursor.execute(sq_query)
        print(sq_query)
        regiondata = cursor.fetchall()
        connection.close()
        cursor.close()        
        return render_template('dataloader.html',raindata=raindata,regiondata=regiondata)
       




@app.route('/fetcluster', methods =  ['GET','POST'])
def fetcluster():
    print("request :"+str(request), flush=True)
    if request.method == 'GET':
        
        connection = mysql.connector.connect(host='localhost',database='rainfalldb',user='root',password='')
        cursor = connection.cursor()
    
        #Fetch the rainfeed
        

        sq_query="SELECT Year,(SUM(JAN)+SUM(FEB)+SUM(MAR)+SUM(APR)+SUM(MAY)+SUM(JUN)+SUM(JUL)+SUM(AUG)+SUM(SEP)+SUM(OCT)+SUM(NOV)+SUM(DECEM)) AS Total FROM rainfalldata GROUP BY Year"
        cursor.execute(sq_query)
        print(sq_query)
        raindata = cursor.fetchall()


        sq_query="SELECT Distinct SUBDIVISION from rainfalldata"
        cursor.execute(sq_query)
        print(sq_query)
        regiondata = cursor.fetchall()



        reg = request.args['reg']

        sq_query="SELECT Year,(SUM(JAN)+SUM(FEB)+SUM(MAR)+SUM(APR)+SUM(MAY)+SUM(JUN)+SUM(JUL)+SUM(AUG)+SUM(SEP)+SUM(OCT)+SUM(NOV)+SUM(DECEM)) AS Total FROM rainfalldata where SUBDIVISION='"+reg+"' and YEAR in (select distinct YEAR from rainfalldata order by CONVERT(YEAR,Decimal)) GROUP BY Year DESC Limit 6"
        cursor.execute(sq_query)
        print(sq_query)
        regraindata = cursor.fetchall()
        connection.close()
        cursor.close()        
        return render_template('dataloader.html',raindata=raindata,regiondata=regiondata,regraindata=regraindata,reg=reg)






@app.route('/predictrain', methods =  ['GET','POST'])
def predictrain():
    print("request :"+str(request), flush=True)
    if request.method == 'GET':
        
        connection = mysql.connector.connect(host='localhost',database='rainfalldb',user='root',password='')
        cursor = connection.cursor()
    
        #Fetch the rainfeed
        

        sq_query="SELECT Year,(SUM(JAN)+SUM(FEB)+SUM(MAR)+SUM(APR)+SUM(MAY)+SUM(JUN)+SUM(JUL)+SUM(AUG)+SUM(SEP)+SUM(OCT)+SUM(NOV)+SUM(DECEM)) AS Total FROM rainfalldata GROUP BY Year"
        cursor.execute(sq_query)
        print(sq_query)
        raindata = cursor.fetchall()


        sq_query="SELECT Distinct SUBDIVISION from rainfalldata"
        cursor.execute(sq_query)
        print(sq_query)
        regiondata = cursor.fetchall()



        reg = request.args['reg']
        ama=reg

        sq_query="SELECT Year,(SUM(JAN)+SUM(FEB)+SUM(MAR)+SUM(APR)+SUM(MAY)+SUM(JUN)+SUM(JUL)+SUM(AUG)+SUM(SEP)+SUM(OCT)+SUM(NOV)+SUM(DECEM)) AS Total FROM rainfalldata where SUBDIVISION='"+reg+"' and YEAR in (select distinct YEAR from rainfalldata order by CONVERT(YEAR,Decimal)) GROUP BY Year DESC Limit 6"
        cursor.execute(sq_query)
        print(sq_query)
        regraindata = cursor.fetchall()


        sq_query="SELECT JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DECEM FROM rainfalldata where SUBDIVISION='"+reg+"' and YEAR in (select distinct YEAR from rainfalldata order by CONVERT(YEAR,Decimal)) GROUP BY Year DESC"
        cursor.execute(sq_query)
        print(sq_query)
        datas = cursor.fetchall()

        sq_query="select distinct YEAR from rainfalldata order by CONVERT(YEAR,Decimal) DESC limit 6"
        cursor.execute(sq_query)
        print(sq_query)
        years = cursor.fetchall()
        print('----------------------------------------------------------')

        pyear=[]
        for i in range(len(years)):
            sq_query="select JAN,FEB,MAR,APR,MAY,JUN,JUL,AUG,SEP,OCT,NOV,DECEM FROM rainfalldata where SUBDIVISION='"+reg+"' and YEAR='"+str(years[i][0])+"'"
            print(sq_query)
            cursor.execute(sq_query)
            tempdata = cursor.fetchall()
            for row in tempdata:
                pyear.append(row)


        print(pyear)
        print(pyear[0][1])
        predictval=[]
        predictval1=[]
        predictval2=[]
        predictval.append(float(pyear[0][0])*mon1)
        predictval.append(float(pyear[0][1])*mon1)
        predictval.append(float(pyear[0][2])*mon1)
        predictval.append(float(pyear[0][3])*mon1)
        predictval.append(float(pyear[0][4])*mon1)
        predictval.append(float(pyear[0][5])*mon1)
        predictval.append(float(pyear[0][6])*mon1)
        predictval.append(float(pyear[0][7])*mon1)
        predictval.append(float(pyear[0][8])*mon1)
        predictval.append(float(pyear[0][9])*mon1)
        predictval.append(float(pyear[0][10])*mon1)
        predictval.append(float(pyear[0][11])*mon1)





        
        predictval1.append(float(pyear[0][0])*mon6)
        predictval1.append(float(pyear[0][1])*mon6)
        predictval1.append(float(pyear[0][2])*mon6)
        predictval1.append(float(pyear[0][3])*mon6)
        predictval1.append(float(pyear[0][4])*mon6)
        predictval1.append(float(pyear[0][5])*mon6)
        predictval1.append(float(pyear[0][6])*mon6)
        predictval1.append(float(pyear[0][7])*mon6)
        predictval1.append(float(pyear[0][8])*mon6)
        predictval1.append(float(pyear[0][9])*mon6)
        predictval1.append(float(pyear[0][10])*mon6)
        predictval1.append(float(pyear[0][11])*mon6)



        
        predictval2.append(float(pyear[0][0])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][1])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][2])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][3])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][4])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][5])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][6])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][7])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][8])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][9])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][10])*((mon1+mon6)/2))
        predictval2.append(float(pyear[0][11])*((mon1+mon6)/2))
        #for i in range(len(pyear[i])):
            #for j in range(12):
                #if i%2==0:
                    #predictval.append(float(pyear[i][j])*mon1)
                #else:
                    #print('a')
                    #predictval.append(float(pyear[i][j])*mon6)
                    
        #print('--------------------------------------')                   
        #print(predictval)

        # Test simple linear regression
        dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
        rmse = evaluate_algorithm(dataset, simple_linear_regression)
        print('RMSE: %.3f' % (rmse))

        from sklearn.svm import SVR
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error

        data = pd.read_csv("rainfall in india 1901-2015.csv",sep=",")
        data = data.fillna(data.mean())
        data.info()
        division_data = np.asarray(data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
           'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])

        X = None; y = None
        for i in range(division_data.shape[1]-3):
            if X is None:
                X = division_data[:, i:i+3]
                y = division_data[:, i+3]
            else:
                X = np.concatenate((X, division_data[:, i:i+3]), axis=0)
                y = np.concatenate((y, division_data[:, i+3]), axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        

        # SVM model
        clf = SVR(gamma='auto', C=0.1, epsilon=0.2)
        clf.fit(X_train, y_train) 
        y_pred = clf.predict(X_test)
        print (mean_absolute_error(y_test, y_pred))


        from sklearn import linear_model

        # linear model
        reg1 = linear_model.ElasticNet(alpha=0.5)
        reg1.fit(X_train, y_train)
        
        #test 2005
        temp = data[['SUBDIVISION','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
               'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == 2005]

        data_2005 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
               'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == reg])

        X_year_2005 = None; y_year_2005 = None
        for i in range(data_2005.shape[1]-3):
            if X_year_2005 is None:
                X_year_2005 = data_2005[:, i:i+3]
                y_year_2005 = data_2005[:, i+3]
            else:
                X_year_2005 = np.concatenate((X_year_2005, data_2005[:, i:i+3]), axis=0)
                y_year_2005 = np.concatenate((y_year_2005, data_2005[:, i+3]), axis=0)

        #test 2010
        temp = data[['SUBDIVISION','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
               'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == 2010]

        data_2010 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
               'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == reg])

        X_year_2010 = None; y_year_2010 = None
        for i in range(data_2010.shape[1]-3):
            if X_year_2010 is None:
                X_year_2010 = data_2010[:, i:i+3]
                y_year_2010 = data_2010[:, i+3]
            else:
                X_year_2010 = np.concatenate((X_year_2010, data_2010[:, i:i+3]), axis=0)
                y_year_2010 = np.concatenate((y_year_2010, data_2010[:, i+3]), axis=0)

        #test 2015
        temp = data[['SUBDIVISION','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
               'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == 2015]

        data_2015 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
               'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == reg])

        print('--------------------------------------------------')

        print(data_2015)

        X_year_2015 = None; y_year_2015 = None
        for i in range(data_2015.shape[1]-3):
            if X_year_2015 is None:
                X_year_2015 = data_2015[:, i:i+3]
                y_year_2015 = data_2015[:, i+3]
            else:
                X_year_2015 = np.concatenate((X_year_2015, data_2015[:, i:i+3]), axis=0)
                y_year_2015 = np.concatenate((y_year_2015, data_2015[:, i+3]), axis=0)


        #2005
        y_year_pred_2005 = reg1.predict(X_year_2005)

        #2010
        y_year_pred_2010 = reg1.predict(X_year_2010)
            
        #2015
        y_year_pred_2015 = reg1.predict(X_year_2015)


       
        print ("MEAN 2005")
        print (np.mean(y_year_2005),np.mean(y_year_pred_2005))
        print ("Standard deviation 2005")
        print (np.sqrt(np.var(y_year_2005)),np.sqrt(np.var(y_year_pred_2005)))


        print ("MEAN 2010")
        print (np.mean(y_year_2010),np.mean(y_year_pred_2010))
        print ("Standard deviation 2010")
        print (np.sqrt(np.var(y_year_2010)),np.sqrt(np.var(y_year_pred_2010)))


        print ("MEAN 2015")
        print (np.mean(y_year_2015),np.mean(y_year_pred_2015))
        print ("Standard deviation 2015")
        print (np.sqrt(np.var(y_year_2015)),np.sqrt(np.var(y_year_pred_2015)))
        svma=Processor.SvmAccuracy()
        lra=Processor.LRccuracy()

        
        
        
        connection.close()
        cursor.close()        
        return render_template('dataloader.html',raindata=raindata,regiondata=regiondata,regraindata=regraindata,reg=ama,predictval=predictval,predictval1=predictval1,predictval2=predictval2,svma=svma,lra=lra)




       


@app.route('/planning')
def planning():
    connection = mysql.connector.connect(host='localhost',database='croppredflask',user='root',password='')
    sql_select_Query = "select * from cropdataset"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()


   
    
    return render_template('planning.html', data=data)



# Calculate root mean squared error
def rmse_metric(actual, predicted):
	sum_error = 0.0
	for i in range(len(actual)):
		prediction_error = predicted[i] - actual[i]
		sum_error += (prediction_error ** 2)
	mean_error = sum_error / float(len(actual))
	return sqrt(mean_error)

# Evaluate regression algorithm on training dataset
def evaluate_algorithm(dataset, algorithm):
	test_set = list()
	for row in dataset:
		row_copy = list(row)
		row_copy[-1] = None
		test_set.append(row_copy)
	predicted = algorithm(dataset, test_set)
	print(predicted)
	actual = [row[-1] for row in dataset]
	rmse = rmse_metric(actual, predicted)
	return rmse

# Calculate the mean value of a list of numbers
def mean(values):
	return sum(values) / float(len(values))

# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
	covar = 0.0
	for i in range(len(x)):
		covar += (x[i] - mean_x) * (y[i] - mean_y)
	return covar

# Calculate the variance of a list of numbers
def variance(values, mean):
	return sum([(x-mean)**2 for x in values])

# Calculate coefficients
def coefficients(dataset):
	x = [row[0] for row in dataset]
	y = [row[1] for row in dataset]
	x_mean, y_mean = mean(x), mean(y)
	b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
	b0 = y_mean - b1 * x_mean
	return [b0, b1]

# Simple linear regression algorithm
def simple_linear_regression(train, test):
	predictions = list()
	b0, b1 = coefficients(train)
	for row in test:
		yhat = b0 + b1 * row[0]
		predictions.append(yhat)
	return predictions




@app.route('/forecast')
def forecast():
    g = geocoder.ip('me')
    print(g.latlng[0])
    print(g.latlng[1])
    print(g)
    
    abc=str(g[0])
    xyz=abc.split(', ')
    print(xyz[0][1:])
    print(xyz[1])
    loc=xyz[0][1:]+", "+xyz[1]
    connection = mysql.connector.connect(host='localhost',database='croppredflask',user='root',password='')
    sql_select_Query = "select * from cropdataset where Area='Nanjangud' and (DYear='2018' or DYear='2019')"
    cursor = connection.cursor()
    cursor.execute(sql_select_Query)
    data = cursor.fetchall()
    connection.close()
    cursor.close()  
    
    return render_template('forecast.html', data=data,glat=g.latlng[0],glon=g.latlng[1],curloc=loc)



if __name__ == '__main__':
    UPLOAD_FOLDER = 'D:/Upload'
    app.secret_key = "secret key"
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run()
