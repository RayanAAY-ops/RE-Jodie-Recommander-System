


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
          learning_rate,
          n_epochs,
          lambda_u,
          lambda_i,
          device,
          ):
  losses_train = []
  losses_valid = []
  all_user_indexes = set(range(0,7047))
  all_item_indexes = set(range(0,98))

  initial_user_embedding = F.normalize(torch.rand(128).to(device), dim=0) # the initial user and item embeddings are learned during training as well
  initial_item_embedding = F.normalize(torch.rand(128).to(device), dim=0)
  optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-5)
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(name)#, param.data)
  print("Training...")
  for e in range(n_epochs):
    optimizer.zero_grad()
    l1,l2=[],[]
    U = initial_user_embedding.repeat(7047, 1) # initialize all users to the same embedding 
    I = initial_item_embedding.repeat(98, 1) # initialize all items to the same embedding
    train_err = 0
    #print("END EPOCH : Users Embeddings after update EPOCH 2 \n")
    #print(I)
    for (_,rows),_ in zip(t_batches.items(),tqdm(range(len(t_batches)), position=0, leave=True)):
      optimizer.zero_grad()
      users_idx,items_idx = extractItemUserId(data,rows)
      l1.append(users_idx)
      l2.append(items_idx)

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

      # Return loss value between the predicted embedding "j_tilde" and the real past item embedding j_true
      loss = MSELoss()(j_tilde,j_true)#.detach()
      loss += regularizer(user_embedding.detach(),future_user_embedding,lambda_u,
                            item_embedding.detach(),future_item_embedding,lambda_i
                            )
      loss += CrossEntropyLoss(weight_ratio_train)(U_pred_state,state_label)
      loss.backward()
      train_err += loss.item()
      optimizer.step()

      users_list = all_user_indexes.difference(set(list(dict.fromkeys(sum(l1,[])))))

      items_list = all_item_indexes.difference(set(list(dict.fromkeys(sum(l2,[])))))
      
    #  U[users_list] = initial_user_embedding.repeat(len(users_list),1)
    #  I[items_list] = initial_item_embedding.repeat(len(items_list),1)
      U[users_idx] = future_user_embedding.detach()
      I[items_idx] = future_item_embedding.detach()
    #print("END EPOCH : EMbeddings after update \n")
    #print(initial_item_embedding)
     # l1 = list(dict.fromkeys(sum(l1,[])))
     # l2 = list(dict.fromkeys(sum(l2,[])))
    print(U)
    y, pred,_,_,auc,valid_err = test_rodie(valid_data,weight_ratio_valid,U.detach().clone(), I.detach().clone(), data, model, device)
    print(U)

    losses_train.append(train_err/len(train_interactions))
    losses_valid.append(valid_err/(len(data)-len(train_interactions)))

    print("validation interactions {}".format(len(data)-len(train_interactions)))
    print("Epoch {} Train Loss {}".format(e,train_err))
    print("Epoch {} Validation Loss {} , AUC Score {}".format(e,valid_err,auc))

    if e%2 ==0:
      torch.save(model.state_dict(), "model_ep{}".format(e))
    print("Saving the model ...")

    if e == n_epochs - 1:
      torch.save(model.state_dict(), "modelFinal_ep{}".format(e))
  return model,U,I,losses_train,losses_valid

