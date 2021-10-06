import numpy as np # used for handling numbers
import pandas as pd # used for handling the dataset
from sklearn.preprocessing import Normalizer

from sklearn.impute import SimpleImputer # used for handling missing data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder # used for encoding categorical data
from sklearn.model_selection import train_test_split # used for splitting training and testing data
from sklearn.preprocessing import StandardScaler # used for feature scaling


from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



import numpy as np
import os
import csv #reading csv

def compute_euclidean_distance(point, centroid):
    return np.sqrt(np.sum((point - centroid)**2))

def assign_label_cluster(distance, data_point, centroids):
    index_of_minimum = min(distance, key=distance.get)
    return [index_of_minimum, data_point, centroids[index_of_minimum]]

def compute_new_centroids(cluster_label, centroids):
    return np.array(cluster_label + centroids)/2



def create_centroids():
    centroids = []
    centroids.append([5.0, 0.0])
    centroids.append([45.0, 70.0])
    centroids.append([50.0, 90.0])
    return np.array(centroids)



def plot_graphs(groundtruth,prediction,title):        
    N = 9
    ind = np.arange(N)  # the x locations for the groups
    width = 0.27       # the width of the bars

    fig = plt.figure()
    fig.suptitle(title, fontsize=12)
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, groundtruth, width, color='r')
    rects2 = ax.bar(ind+width, prediction, width, color='g')

    ax.set_ylabel("Amount of rainfall")
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('APR', 'MAY', 'JUN', 'JUL','AUG', 'SEP', 'OCT', 'NOV', 'DEC') )
    ax.legend( (rects1[0], rects2[0]), ('Ground truth', 'Prediction') )

#     autolabel(rects1)
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')
#     autolabel(rects2)

    plt.show()



if __name__ == "__main__":
    #filename = os.path.dirname(__file__) + "\kddtest1.xls"
    #data_points = np.genfromtxt(filename, delimiter=",")
    
    centroids = create_centroids()
    total_iteration = 100
    dataset_filename = 'rainfall.csv'
    #dataset = pd.read_csv(dataset_filename)
    #print(dataset)
    #print(dataset[0])

    fn = 'rainfall.csv'
    fields = [] 
    rows = []

    with open(fn, 'r') as csvfile:
            # creating a csv reader object 
            csvreader = csv.reader(csvfile)  
  
            # extracting each data row one by one 
            for row in csvreader:
                rows.append(row)
                #print(row)



    print("Work Done")


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
    print('ssssssssssssssssssssssssssssssssssssssssssssssssss')
    print (mean_absolute_error(y_test, y_pred))


    from sklearn import linear_model

    # linear model
    reg = linear_model.ElasticNet(alpha=0.5)
    reg.fit(X_train, y_train)
    
    #test 2005
    temp = data[['SUBDIVISION','JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
           'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == 2005]

    data_2005 = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
           'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == 'TELANGANA'])

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
           'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == 'TELANGANA'])

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
           'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[temp['SUBDIVISION'] == 'TELANGANA'])

    X_year_2015 = None; y_year_2015 = None
    for i in range(data_2015.shape[1]-3):
        if X_year_2015 is None:
            X_year_2015 = data_2015[:, i:i+3]
            y_year_2015 = data_2015[:, i+3]
        else:
            X_year_2015 = np.concatenate((X_year_2015, data_2015[:, i:i+3]), axis=0)
            y_year_2015 = np.concatenate((y_year_2015, data_2015[:, i+3]), axis=0)


    #2005
    y_year_pred_2005 = reg.predict(X_year_2005)

    #2010
    y_year_pred_2010 = reg.predict(X_year_2010)
        
    #2015
    y_year_pred_2015 = reg.predict(X_year_2015)


   
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

    plot_graphs(y_year_2005,y_year_pred_2005,"Year-2005")
    plot_graphs(y_year_2010,y_year_pred_2010,"Year-2010")
    plot_graphs(y_year_2015,y_year_pred_2015,"Year-2015")

        



    for i in range(len(rows)):
        #print(rows[i][0])
        if(rows[i][0]=="SOUTH INTERIOR KARNATAKA"):
            print(rows[i][1])



    

    #dataset = pd.read_csv(dataset_filename, sep=',', decimal='.', header=None)



    # Splitting the attributes into independent and dependent attributes
    #X = dataset.iloc[:, 1:19].values # attributes to determine dependent variable / Class
    #Y = dataset.iloc[:, 0].values # dependent variable / Class


    #print(X)
    #print(Y)


    #scaler = Normalizer().fit(X)
    #trainX = scaler.transform(X)
    #print(trainX)


    #print(dataset[0])
    
    
