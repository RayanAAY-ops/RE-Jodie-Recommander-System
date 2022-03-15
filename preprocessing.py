import numpy as np
import pandas as pd
from collections import defaultdict

def extract_data_mooc():

  features = pd.read_csv("/content/act-mooc/mooc_action_features.tsv",sep="\t")
  labels = pd.read_csv("/content/act-mooc/mooc_action_labels.tsv",sep="\t")
  users = pd.read_csv("/content/act-mooc/mooc_actions.tsv",sep="\t")

  print("features columns {}\n".format(features.columns))
  print("labels columns {}".format(labels.columns)) 
  print("users columns {}".format(users.columns)) 

  join1 = labels.merge(users,left_index=True,right_index=True)#,on="ACTIONID")

  join2 = join1.merge(features,left_index=True,right_index=True)

  join2 = join2[["USERID","TARGETID","TIMESTAMP","LABEL","FEATURE0","FEATURE1","FEATURE2","FEATURE3"]]
  join2.columns = ["user_id","item_id","timestamp","state_label","f1","f2","f3","f4"  ]
  join2.to_csv("data/mooc.csv",index=False)
  join2.index.name ="ACTIONID"
  return join2
  
def extractItemUserId(data,idx):
    users_idx = data[idx,0]
    items_idx = data[idx,1]
    return [int(x) for x in users_idx], [int(x) for x in items_idx]

def extractFeatures(data,idx):
    state_label = data[idx,3]
    delta_u = data[idx,4]
    delta_i = data[idx,5]
    f = data[idx,8:]
    return state_label,delta_u,delta_i,f

def extractNextStateItem(data,idx):
    next_state  = data[idx,7]
    next_item  = data[idx,6]
    return next_state,next_item



### This function enables to extract the next label of each user at t+1 
def extractNextUserState(data_pandas):
  data_sorted = data_pandas.sort_values(["user_id","timestamp"])[["user_id","state_label"]]
  data_sorted['next_state_label'] = pd.Series()
  data_numpy = data_sorted.values
  for u in range(len(data_numpy)):
    try:
      if (data_numpy[u,0] != data_numpy[u+1,0]):
        data_numpy[u,2] = -1
      else:
        data_numpy[u,2] = data_numpy[u+1,1]
    except IndexError as e:
      data_numpy[u,2] = -1
      pass
  data_sorted['next_state_label'] = data_numpy[:,-1]
  return data_sorted.reindex(data_pandas.index)['next_state_label'].astype('int')

### This function enables to extract the future item the user will interact with in the near future 
def UserNextInteraction(data_pandas):
  sort_data = data_pandas.sort_values(["user_id","timestamp"])[["user_id","item_id","timestamp"]]
  sort_data["nextItemInteraction"] = pd.Series()
  sort_data_numpy = sort_data.values
  for u in range(len(sort_data_numpy)):
    try:
      if (sort_data_numpy[u,0] != sort_data_numpy[u+1,0]):
        sort_data_numpy[u,3] = -1
      else:
        sort_data_numpy[u,3] = sort_data_numpy[u+1,1]
    except IndexError as e:
        sort_data_numpy[u,3] = -1
        pass

  sort_data['nextItemInteraction'] = sort_data_numpy[:,-1]
  return sort_data.reindex(data_pandas.index)['nextItemInteraction'].astype('int')

### This function is used to compute delta_i and delta_u respectively for items and users ###
# using the parameter "entity", you can choose to compute delta_i -> for item or delta_u -> for user
def delta(data_pandas,entity):
  sort_data = data_pandas.sort_values([entity,"timestamp"])[[entity,"timestamp"]]
  sort_data['delta_{}'.format(entity)] = pd.Series()
  sort_data_numpy = sort_data.values

  for i in range(len(sort_data_numpy)):
    if i==0:
      sort_data_numpy[i,2] = 0
    try:
      if (sort_data_numpy[i,0]!=sort_data_numpy[i+1,0]):
        sort_data_numpy[i+1,2] = 0
      else:
        sort_data_numpy[i+1,2] = sort_data_numpy[i+1,1] - sort_data_numpy[i,1]

    except IndexError as e:
        pass
  
  print("delta {}".format(entity))

  sort_data['delta_{}'.format(entity)] = sort_data_numpy[:,-1]
  return sort_data.reindex(data_pandas.index)['delta_{}'.format(entity)].astype('int')





def t_batch_update(data):
  # Random Data to test
  print("T-Batch start...")
  user_seq = data['user_id'].values
  item_seq = data['item_id'].values
  nb_interaction = len(user_seq)

  tbatch_id_user = defaultdict(lambda: -1)
  tbatch_id_item = defaultdict(lambda: -1)
  tbatch_interaction = defaultdict(list)

  print("Number of interaction = {}".format(nb_interaction))
  for j in range(nb_interaction):  # on parcours toutes les interactions
    id_user = user_seq[j]
    id_item = item_seq[j]

    index_tbatch = max(tbatch_id_user[id_user], tbatch_id_item[id_item]) + 1

    tbatch_id_user[id_user] = index_tbatch
    tbatch_id_item[id_item] = index_tbatch

    tbatch_interaction[index_tbatch].append(j)

  print("T-Batch ends !")
  return tbatch_interaction