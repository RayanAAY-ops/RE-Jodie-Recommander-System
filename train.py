
from torch.nn import MSELoss, HuberLoss,L1Loss,CrossEntropyLoss
from preprocessing import *
import torch
from  torch import nn
from torch.nn import RNNCell
from torch.nn.functional import one_hot
import math
from tqdm import tqdm
from torch.nn import MSELoss, HuberLoss,L1Loss,CrossEntropyLoss
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from preprocessing import *
from model import *
from tqdm import tqdm

from torch.nn import MSELoss, HuberLoss,L1Loss,CrossEntropyLoss
from preprocessing import *
import torch
from  torch import nn
from torch.nn import RNNCell
from torch.nn.functional import one_hot
import math
from tqdm import tqdm
from torch.nn import MSELoss, HuberLoss,L1Loss,CrossEntropyLoss
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from preprocessing import *
from model import *
from tqdm import tqdm


from torch.utils.data import DataLoader
def dynamic_embedding(data,embedding_dim):
        num_users = len(torch.unique(data[:,0]))
        num_items = len(torch.unique(data[:,1]))
        dynamic_users_embedding = F.normalize(torch.rand(embedding_dim),dim=0).repeat(num_users,1)
        dynamic_items_embedding = F.normalize(torch.rand(embedding_dim),dim=0).repeat(num_items+1,1)

        print("Initialisation of dynamic embedding... Done !")
        print("Dynamic Embedding shape : Users {}, \t Items {}".format(list(dynamic_users_embedding.size()),list(dynamic_items_embedding.size())))

        return dynamic_users_embedding,dynamic_items_embedding



def regularizer(actual_user_embedding,future_user_embedding,lambda_u,
                               actual_item_embedding,future_item_embedding,lambda_i
                               ):
    u_regularization_loss =  MSELoss()(actual_user_embedding,future_user_embedding)
    i_regularization_loss =  MSELoss()(actual_item_embedding,future_item_embedding)
    return lambda_u* u_regularization_loss + lambda_i* i_regularization_loss 


def train_rodie(t_batches,
          data,
          valid_data,
          train_interactions,
          weight_ratio_train,
          weight_ratio_valid,
          model,
          optimizer,
          learning_rate,
          n_epochs,
          lambda_u,
          lambda_i,
          device,
          ):
  losses_train = []
  losses_valid = []
  print("Training...")
  for e in range(n_epochs):
    train_err = 0
    U,I = dynamic_embedding(data,model.embedding_dim)  # Initial dynamic embedding
 # U_copy,I_copy = U.clone(), I.clone()
    U = U.to(device)
    I = I.to(device)
    for (_,rows),_ in zip(t_batches.items(),tqdm(range(len(t_batches)), position=0, leave=True)):
      optimizer.zero_grad()
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
      
      # The forward pass of the model : extract dynamic embeddings (user+item ), and predicted user state and predicted item embedding
      future_user_embedding,future_item_embedding,U_pred_state,j_tilde,j_true = model(item_embedding,
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
      
      # Return loss value between the predicted embedding "j_tilde" and the real past item embedding j_true
      
      loss = MSELoss()(j_tilde,j_true)#.detach()
      loss += regularizer(user_embedding.detach(),future_user_embedding,lambda_u,
                            item_embedding.detach(),future_item_embedding,lambda_i
                            )
      loss += CrossEntropyLoss(weight_ratio_train)(U_pred_state,state_label)
      loss.backward()
      train_err += loss.item()
      optimizer.step()

    y, pred,auc,valid_err = test_rodie(valid_data,weight_ratio_valid,U, I, data, model, device)
    losses_train.append(train_err/len(train_interactions))
    losses_valid.append(valid_err/(len(data)-len(train_interactions)))

    print("validation interactions {}".format(len(data)-len(train_interactions)))
    print("Epoch {} Train Loss {}".format(e,losses_train[e]))
    print("Epoch {} Validation Loss {} , AUC Score {}".format(e,losses_valid[e],auc))

    if e%2 ==0:
      torch.save(model.state_dict(), "model_ep{}".format(e))

    print("Saving the model ...")
    torch.save(model.state_dict(), "modelFinal_ep{}".format(e))
  return model,U,I,losses_train,losses_valid








