#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from datetime import date, timedelta


# In[2]:


#Uploading each file, parsing dates and using Scheme modifications to reduce size of file
CasinoA = pd.read_excel('CasinoA.xlsx',                        parse_date = ('Registration_Date','First_Deposit_Date','Last_Deposit_Date','Last_Login','birthdate'),                       dtype = {'Gender':'category', 'Bonus_Opt_Out':'category', 'VIP_Status':'category', 'Country':'category'})                    
CasinoD = pd.read_excel('CasinoD.xlsx',                        parse_date = ('Registration_Date','First_Deposit_Date','Last_Deposit_Date','Last_Login','birthdate'),                       dtype = {'Gender':'category', 'Bonus_Opt_Out':'category', 'VIP_Status':'category', 'Country':'category'})                        
CasinoE = pd.read_excel('CasinoE.xlsx',                        parse_date = ('Registration_Date','First_Deposit_Date','Last_Deposit_Date','Last_Login','birthdate'),                       dtype = {'Gender':'category', 'Bonus_Opt_Out':'category', 'VIP_Status':'category', 'Country':'category'})
CasinoH = pd.read_excel('CasinoH.xlsx',                        parse_date = ('Registration_Date','First_Deposit_Date','Last_Deposit_Date','Last_Login','birthdate'),                       dtype = {'Gender':'category', 'Bonus_Opt_Out':'category', 'VIP_Status':'category', 'Country':'category'})
CasinoB = pd.read_excel('CasinoB.xlsx',                        parse_date = ('Registration_Date','First_Deposit_Date','Last_Deposit_Date','Last_Login','birthdate'),                       dtype = {'Gender':'category', 'Bonus_Opt_Out':'category', 'VIP_Status':'category', 'Country':'category'})
CasinoC = pd.read_excel('CasinoC.xlsx',                        parse_date = ('Registration_Date','First_Deposit_Date','Last_Deposit_Date','Last_Login','birthdate'),                       dtype = {'Gender':'category', 'Bonus_Opt_Out':'category', 'VIP_Status':'category', 'Country':'category'})
CasinoF = pd.read_excel('CasinoF.xlsx',                        parse_date = ('Registration_Date','First_Deposit_Date','Last_Deposit_Date','Last_Login','birthdate'),                       dtype = {'Gender':'category', 'Bonus_Opt_Out':'category', 'VIP_Status':'category', 'Country':'category'})
CasinoG = pd.read_excel('CasinoG.xlsx',                           parse_date = ('Registration_Date','First_Deposit_Date','Last_Deposit_Date','Last_Login','birthdate'),                       dtype = {'Gender':'category', 'Bonus_Opt_Out':'category', 'VIP_Status':'category', 'Country':'category'})


# In[3]:


# Union all the seperate brands into one file for further analysis
allcasinos = pd.concat([CasinoA,CasinoD,CasinoE,CasinoH,CasinoB,CasinoC,CasinoF,CasinoG])


# In[4]:


# Check to see number of Columns and Rows
allcasinos.shape


# In[5]:


#Created Column to specify age
allcasinos["Age"] = np.round((pd.Timestamp.today() - allcasinos["birthdate"])/np.timedelta64(1,'Y'))


# In[6]:


#Categorised Age into different bins
bins = [0, 18, 30, 40, 50, 60, 70]
names = ['0-17', '18-30', '31-40', '41-50', '51-60','61-70','70+']

d = dict(enumerate(names, 1))

allcasinos['AgeRange'] = np.vectorize(d.get)(np.digitize(allcasinos['Age'], bins))


# In[7]:


#Define First Deposit date as Converted user to enable usage of data for visualation
allcasinos['Converter'] =  np.where(allcasinos['First_Deposit_Date'] >='2010-01-01', 'yes', "")


# In[8]:


#Created Column to determine date difference from registration to FTD(First time deposit)
allcasinos["REG_to_FTD"] = np.round((allcasinos["First_Deposit_Date"] - allcasinos["Registration_Date"])/np.timedelta64(1,'D'))


# In[9]:


#Created Column to determine date diff from First Deposit Date to Last Login Date
#This parameter will be used to determine the life span of a converter
allcasinos["Lifespan"] = np.round((allcasinos["Last_Login"] - allcasinos["First_Deposit_Date"])/np.timedelta64(1,'D'))


# In[10]:


#Display All Column names
allcasinos.columns


# In[11]:


#Obtaining a breakdown of all games and assessing which games need to be reframed as table games
unique_games = allcasinos["Favorite_game"].unique()
print(unique_games)


# In[12]:


#Creating a function to reframe respective favourite game to either Table or Slot games for further research
def table_slots(x):
    if x == "American Blackjack":
        output = 'Table'
    elif x == "American (US) Blackjack":
        output = 'Table'
    elif x == 'VIP Blackjack':
        output = 'Table'
    elif x == 'VIP Blackjack 2':
        output = 'Table'
    elif x == 'Blackjack Surrender':
        output = 'Table'
    elif x == 'BlackJack':
        output = 'Table'
    elif x == 'Micro Roulette':
        output = 'Table'
    elif x == 'Single Deck Blackjack':
        output = 'Table'
    elif x == 'American Roulette':
        output = 'Table'
    elif x == 'Roulette':
        output = 'Table'
    elif x == 'European Roulette':
        output = 'Table'
    elif x == 'VIP Roulette':
        output = 'Table'
    elif x == 'Lucky Spin European Roulette':
        output = 'Table'
    elif x == 'Zero Spin Roulette':
        output = 'Table'
    elif x == 'Blackjack Neon':
        output = 'Table'
    elif x == 'Blackjack 3':
        output = 'Table'
    elif x == "Premium Roulette":
        output = 'Table'
    elif x == "Baccarat":
        output = 'Table'
    elif x == "Salon Prive Roulette":
        output = 'Table'
    elif x == "Roulette Pro":
        output = 'Table'
    elif x == "Immersive Roulette":
        output = 'Table'
    elif x == "Casino Hold'em Poker":
        output = 'Table'
    elif x == "Speed Auto Roulette":
        output = 'Table'
    elif x == "Neon BJ Single Deck":
        output = 'Table'
    elif x == "Neon Roulette":
        output = 'Table'
    else:
        output = 'Slot'
    return output


# In[13]:


#Applying the function
allcasinos['Favorite_game'] = allcasinos['Favorite_game'].apply(table_slots)


# In[14]:


#Assessing the function results
allcasinos["Favorite_game"].value_counts()


# In[15]:


#Dataset distribution to be visualised - Keeping code
subject = ['Registration_Date','First_Deposit_Date']
dataset = allcasinos.groupby('AgeRange')[subject].count()

index = np.arange(len(subject))
total_count = np.arange(0,80000,5000)
print(dataset.T)


# In[16]:


#Preparing Data for Age Range Distribution
labels = ['18-30', '31-40', '41-50', '51-60','61-70' ,'70+']
Registration_Date = [405380, 610052, 240726, 94892, 39273, 23052]
First_Deposit_Date = [17061, 22971, 14634, 9035, 3616,1292]

x = np.arange(len(labels))  
width = 0.45

#Plotting Graph
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, Registration_Date, width, label='Reg')
rects2 = ax.bar(x + width/2, First_Deposit_Date, width, label='Con')

ax.set_ylabel('Total')
ax.set_title('Registrations vs Conversions Age Dist.')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#Plotting Text, labels within Graph
def autolabel(rects):
      for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4), 
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()


# In[17]:


#Converion rate table for AgeRange
dfreg = allcasinos.groupby('AgeRange')['Registration_Date'].count()  

dfcon = allcasinos.groupby('AgeRange')['First_Deposit_Date'].count()  

dftotal = pd.concat([dfreg,dfcon], axis = 'columns', sort=False)
dftotal['Conversion_Rate_%'] = (dftotal['First_Deposit_Date']/dftotal['Registration_Date']) * 100
dftotal.head()


# In[18]:


# Preparing the Pie Chart for Gender distribution for Registrations
df_pie=allcasinos.groupby("gender")["playerId"].count().to_frame().rename(columns={"playerId":"Count"}).sort_values("Count", ascending=False)

df_pie.reset_index(drop=False, inplace=True)
df_pie["Label"]=df_pie["gender"]+": "+ df_pie["Count"].astype(str)


# In[19]:


df_pie


# In[20]:


#Plotting Pie Graph
p_labels=df_pie["Label"]
p_size=df_pie["Count"]
p_explode=[0.1 for i in range(len(p_labels))]
plt.pie(p_size, startangle=50, explode=p_explode, shadow=True, autopct='%.f%%', labels=p_labels)
plt.title("Gender Distribution (Registrations)\n",fontsize=15)
plt.axis('equal')
plt.show()


# In[21]:


# Preparing the Pie Chart for Gender distribution for Converions
df_pie2=allcasinos[allcasinos['Converter'] == 'yes'].groupby("gender")["playerId"].count().to_frame().rename(columns={"playerId":"Count"}).sort_values("Count", ascending=False)

df_pie2.reset_index(drop=False, inplace=True)
df_pie2["Label"]=df_pie2["gender"]+": "+ df_pie2["Count"].astype(str)


# In[22]:


df_pie2


# In[23]:


p_labels=df_pie2["Label"]
p_size=df_pie2["Count"]
p_explode=[0.1 for i in range(len(p_labels))]
plt.pie(p_size, startangle=50, explode=p_explode, shadow=True, autopct='%.f%%', labels=p_labels)
plt.title("Gender Distribution (Conversions)\n",fontsize=15)
plt.axis('equal')
plt.show()


# In[24]:


#Multiple Graphs in Same Figure - Object Orientated
# Choosing ggplot style
plt.style.use('ggplot')

# Get the figure and the axes (or subplots)
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(18, 8))

# Prepare data for ax0  
Registration_Country = allcasinos.groupby('Country')['Registration_Date'].count()
x = Registration_Country.index.values
height = Registration_Country.values

# Axes.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)
ax0.bar(x, height, width=0.5, align='center')
ax0.set(title = 'Registrations', xlabel='Countries' , ylabel = 'Total Registrations')

#Second graph
Depositors_Country = allcasinos[allcasinos['Converter'] == 'yes'].groupby('Country')['First_Deposit_Date'].count()

ax1 = Depositors_Country.plot(kind='bar', color="green", fontsize=13);
ax1.set_alpha(0.8)
ax1.set_title("Convertors", fontsize=13)
ax1.set_ylabel("Total Depositors", fontsize=13)
ax1.set_xlabel("Countries",fontsize=13)


# Title the figure
fig.suptitle('Country Counts', fontsize=14, fontweight='bold')


# In[25]:


#Converion rate table for Countries
Country_reg = allcasinos.groupby('Country')['Registration_Date'].count()  

Country_con = allcasinos.groupby('Country')['First_Deposit_Date'].count()  

CCtotal = pd.concat([Country_reg,Country_con], axis = 'columns', sort=False)
CCtotal['Conversion_Rate-%'] = (CCtotal['First_Deposit_Date']/CCtotal['Registration_Date']) * 100
CCtotal.head(10)


# In[26]:


#Graphing Life span on players from different regions
Average_country = np.round(allcasinos.groupby("Country")["Lifespan"].mean())
print(Average_country)

ax = Average_country.plot(kind='bar', figsize=(10,6), color="blue", fontsize=13);
ax.set_alpha(0.8)
ax.set_title("Average Life Span", fontsize=22)
ax.set_ylabel("Average Days", fontsize=15)
ax.set_xlabel("Countries",fontsize=15)
plt.show()


# In[27]:


#Statisical Breakdown of Lifespan of user
Average_country = np.round(allcasinos.groupby("Country")["Lifespan"].mean())
print(Average_country)
Average_country.describe().to_frame()


# In[28]:


gb_accstat = allcasinos[allcasinos['Converter'] == 'yes'].groupby("Account_Status")["playerId"].count().to_frame().sort_values("Account_Status", ascending=False)

ax = gb_accstat.plot(kind='barh', figsize=(10,6), color="indigo", fontsize=13);
ax.set_alpha(0.8)
ax.set_title("User Count", fontsize=22)
ax.set_ylabel("Accountstatus", fontsize=15)
ax.legend(loc='lower right')
plt.show()


# In[29]:


#Distribution of Account status
filt = allcasinos['Converter'] == 'yes'
Acc_status = allcasinos.loc[filt]['Account_Status'].value_counts().to_frame().rename(columns = {"Account_Status":"Number_of_users"})
Acc_status.head(8)


# In[30]:


#Graphing Average days on players from different regions to FTD
Average_country = np.round(allcasinos.groupby("Country")["REG_to_FTD"].mean())
print(Average_country)

ax = Average_country.plot(kind='bar', figsize=(10,6), color="orange", fontsize=13);
ax.set_alpha(0.8)
ax.set_title("Average Days before deposit", fontsize=22)
ax.set_ylabel("Average Days", fontsize=15)
ax.set_xlabel("Countries",fontsize=15)
plt.show()


# In[31]:


#Groupby Analysis of Profitable players
Profit_analysis = allcasinos.groupby(["VIP_Status","Country", "gender", 'Favorite_game']).agg({"Total_Deposit_Amount" : ["sum","mean"],
                               "Age" : ["mean"], "Average_bet" : ["mean"], "Lifespan" : ["mean"]}).dropna()
Profit_analysis.head(50)


# In[ ]:


#In order to see the respective VIP status
Profit_analysis = allcasinos.groupby(["VIP_Status","Country", "gender", 'Favorite_game']).agg({"Total_Deposit_Amount" : ["sum","mean"],
                               "Age" : ["mean"], "Average_bet" : ["mean"], "Lifespan" : ["mean"]}).dropna()
Part_black = Profit_analysis.head(19)
Part_Platinum = Profit_analysis.tail(40)
Part_Gold = Profit_analysis.tail(65)


# In[32]:


#Preparing Pie Chart to show Total Sum deposits Per region
TotalDepSum_country = np.round(allcasinos.groupby("Country")["Total_Deposit_Amount"].sum())
print(TotalDepSum_country)

# Create a Figure and get its axes using subplots
fig, ax = plt.subplots(figsize=(15, 7), subplot_kw=dict(aspect="equal"))

# Prepare data
types = [x for x in (TotalDepSum_country.index.values)]
values = [x for x in (TotalDepSum_country.values)]

# Lambda function
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} )".format(pct, absolute)

# ax.pie
wedges, texts, autotexts = ax.pie(values, explode = (0.05, 0.05, 0.05, 0.05, 0.05, 0.05),                                  autopct=lambda pct: func(pct, values), shadow=False, textprops=dict(color="black"))

ax.legend(wedges, types,
          title="Per Region",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))


ax.set_title("Total Sum deposits Per Region")



plt.setp(autotexts, size=9, weight="bold")

plt.show()

