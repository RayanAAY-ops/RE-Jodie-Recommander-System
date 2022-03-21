
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

def dynamic_embedding(data,embedding_dim):
        num_users = len(torch.unique(data[:,0]))
        num_items = len(torch.unique(data[:,1]))
        dynamic_users_embedding = F.normalize(torch.randn(embedding_dim).repeat(num_users,1),dim=0)#nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')#F.normalize(torch.randn(num_users,embedding_dim))
        dynamic_items_embedding = F.normalize(torch.randn(embedding_dim).repeat(num_items+1,1),dim=0)

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
          weight_ratio,
          model,
          optimizer,
          learning_rate,
          n_epochs,
          lambda_u,
          lambda_i,
          device,

          ):

  U,I = dynamic_embedding(data,model.embedding_dim)  # Initial dynamic embedding
    
  U = U.to(device)
  I = I.to(device)
  print("Training...")
 # U_copy = U.clone().detach()
 # I_copy = I.clone().detach()
  #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
  for e in range(n_epochs):
    l = 0
    
    for (_,rows),_ in zip(t_batches.items(),tqdm(range(len(t_batches)), position=0, leave=True)):
      optimizer.zero_grad()
      users_idx,items_idx = extractItemUserId(data,rows)
      state_label,delta_u,delta_i,f = extractFeatures(data,rows)
      next_item = extractPastItem(data,rows)
      u_static, i_static = model.static_users_embedding[users_idx].detach(), model.static_items_embedding[items_idx].detach()


      user_embedding, item_embedding = U[users_idx], I[items_idx]
      next_item_static_embedding, next_item_dynamic_embedding = model.static_items_embedding[[int(x) for x in next_item]].detach(), I[[int(x) for x in next_item]].detach()

      u_static = u_static.to(device)
      i_static = i_static.to(device)
      f = f.to(device)
      delta_u = delta_u.to(device)
      delta_i = delta_i.to(device)
      state_label = state_label.type(torch.LongTensor).to(device)
      next_item_dynamic_embedding = next_item_dynamic_embedding.to(device)
      next_item_static_embedding = next_item_static_embedding.to(device)
      
      # The forward pass of the model : extract dynamic embeddings (user+item ), and predicted user state and predicted item embedding
      future_user_embedding,future_item_embedding,U_pred_state,j_tilde,j_true = model(item_embedding,
                user_embedding,
                u_static,
                i_static,
                f,
                delta_u,
                delta_i,
                
                next_item_dynamic_embedding,
                next_item_static_embedding)
      # Add the new embedding to the placeholder U and I
      U[users_idx] = future_user_embedding.detach().clone()
      I[items_idx] = future_item_embedding.detach().clone() 
      
      # Return loss value between the predicted embedding "j_tilde" and the real next item embedding j_true
      
      loss = MSELoss()(j_tilde,j_true)#.detach()
      loss += regularizer(user_embedding.detach(),future_user_embedding,lambda_u,
                            item_embedding.detach(),future_item_embedding,lambda_i
                            )
      
      loss += CrossEntropyLoss(weight_ratio)(U_pred_state,state_label)
      #print(I[0])
      loss.backward()
      l += loss.item()
     # torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=100.)
      optimizer.step()

    #scheduler.step(loss)
    print("Epoch {} Loss {}".format(e,l))

  return model,U,I
