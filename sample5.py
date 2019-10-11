import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()

# File created by combining transaction summary,transaction details and products 
dir_path = "//Users/mickey/Desktop/Projects/DNB Digital/R Project/Fabby/ProductRecommendation/Data1/"
fulldata=pd.read_csv(dir_path+'AprioriData.csv')


#filtering the dataset for a 'Customer'
dataset=fulldata.loc[fulldata['customer_id'] == 100001]
dataset= dataset['product_name.y']
print(dataset)

#convert List of transactions List of lists
def extractAsLists(lst): 
    res = [] 
    for el in lst: 
        sub = el.split(',') 
        res.append(sub) 
      
    return(res) 
                 
print(extractAsLists(dataset)) 
dataset = extractAsLists(dataset)
    



te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
# df.to_csv(dir_path+'AprioriDatasampledata.csv',encoding='utf-8',index=False)


from mlxtend.frequent_patterns import apriori

frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets) 


# result = frequent_itemsets[ (frequent_itemsets['length'] == 3) &
#                    (frequent_itemsets['support'] >= 0.1) ]
# print(frequent_itemsets)
# result.to_csv(dir_path+'Aprioriresult.csv',encoding='utf-8',index=False)