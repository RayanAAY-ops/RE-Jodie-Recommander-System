from sklearn.metrics import roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from preprocessing import *
from model import *
from tqdm import tqdm
def test_rodie(t_batches_test,U,I,data,model,device):
  y_true = []
  y_pred = []
  print("Testing...")
  
  for rows,_ in zip(test.index.values.tolist(),tqdm(range(len(test)), position=0, leave=True)):
    users_idx,items_idx = extractItemUserId(data,rows)
    state_label,delta_u,delta_i,f = extractFeatures(data,rows)
    past_item = extractPastItem(data,rows)
    u_static, i_static = model.static_users_embedding[users_idx], model.static_items_embedding[items_idx]
    user_embedding, item_embedding = U[users_idx], I[items_idx]
    past_item_static_embedding, past_item_dynamic_embedding = model.static_items_embedding[[int(x) for x in past_item]], I[[int(x) for x in past_item]]

    u_static = u_static.to(device)
    i_static = i_static.to(device)
    f = f.to(device)
    delta_u = delta_u.to(device)
    delta_i = delta_i.to(device)
    state_label = state_label.type(torch.LongTensor).to(device)
    past_item_dynamic_embedding = past_item_dynamic_embedding.to(device)
    past_item_static_embedding = past_item_static_embedding.to(device)
    
    # The forward pass of the model : extract dynamic embeddings (user+item), and predicted user state and predicted item embedding
    future_user_embedding,future_item_embedding,U_pred_state,_,_ = model(item_embedding,
              user_embedding,
              u_static,
              i_static,
              f,
              delta_u,
              delta_i,
              past_item_dynamic_embedding,
              past_item_static_embedding)
    # Add the new embedding to the placeholder U and I
    U[users_idx] = future_user_embedding.detach()
    I[items_idx] = future_item_embedding.detach()

    y_true.append(state_label.detach().cpu().numpy())
    y_pred.append(U_pred_state.detach().cpu().numpy()[:,1])
    
  y_true=[x.tolist() for x in y_true] # Convert list of numpy array to list of list
  y_true= sum(y_true, [])  # Convert list of list to list

  y_pred=[x.tolist() for x in y_pred]  # Convert list of numpy array to list of list
  y_pred=sum(y_pred, []) # Convert list of list to list
  auc = roc_auc_score(y_true, y_pred)

  return y_true, y_pred, auc

#### TSNE PLOT ####
def plot_tsne(data,embedding,interactions,entity):
  new_df = data.iloc[interactions,:].copy()
  list_index = np.unique(new_df[entity]).tolist()


  data_  = (embedding.detach().cpu().clone()).numpy()[list_index]
  df = pd.DataFrame(data_)

  if entity =="user_id":
    list_of_change = new_df[new_df['state_label'] == 1][entity].values
    df['label'] = np.zeros((len(list_index),1))
    for index, row in df.iterrows():
        for d in list_of_change:
          if index == d:
            df.iloc[index,-1] = 1
  ### TSNE
  tsne = TSNE(2)
  data_tsne = tsne.fit_transform(data_)
  df[['t1','t2']] = data_tsne
  if entity =="user_id":
    sns.scatterplot(data=df, x="t1", y="t2", hue="label",style="label")
  else:
    sns.scatterplot(data=df, x="t1", y="t2")

  return data_tsne





