
import torch 
from model import *
from preprocessing import *
 


def train_rodie(t_batches,
          data,
          U,
          I,
          model,
          optimizer,
          n_epochs,
          lambda_u,
          lambda_i,
          device
          ):
  print("Training...")
 # U_copy = U.clone().detach()
 # I_copy = I.clone().detach()

  for e in range(n_epochs):
    l = 0
    
    for (_,rows) in t_batches.items():
      optimizer.zero_grad()
      users_idx,items_idx = extractItemUserId(data,rows)

      state_label,delta_u,delta_i,f = extractFeatures(data,rows)

      next_state,next_item = extractNextStateItem(data,rows)

      u_static, i_static = model.static_users_embedding[users_idx], model.static_users_embedding[items_idx]

      user_embedding, item_embedding = U[users_idx], I[items_idx]
      next_item_static_embedding, next_item_dynamic_embedding = model.static_users_embedding[[int(x) for x in next_item]], I[[int(x) for x in next_item]]


      item_embedding = item_embedding.to(device)
      user_embedding  = user_embedding.to(device)
      u_static = u_static.to(device)
      i_static = i_static.to(device)
      f = f.to(device)
      delta_u = delta_u.to(device)
      delta_i = delta_i.to(device)
      next_state = next_state.type(torch.LongTensor).to(device)
      next_item_dynamic_embedding = next_item_dynamic_embedding.to(device)
      next_item_static_embedding = next_item_static_embedding.to(device)
      future_user_embedding,future_item_embedding,loss  = model(item_embedding,
                user_embedding,
                u_static,
                i_static,
                f,
                delta_u,
                delta_i,
                next_state,
                next_item_dynamic_embedding,
                next_item_static_embedding) # a revoir

      U[users_idx] = future_user_embedding.clone().detach()
      I[items_idx] = future_item_embedding .clone().detach() 
      loss.backward()
      l += loss.item()
      optimizer.step()
      #with torch.no_grad():   # Vérifier changement de U et I

    print("Epoch {} Loss {}".format(e,l))

  return model,U,I