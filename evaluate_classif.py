def evaluate_classif(model, data_test, tBatches_test, U_test, I_test):

  pred = []  # liste vide pour mettre les prédictions
  true_labels = []
  for batch in tBatches_test:
      l_u,_ = extractItemUserId(batch) # extraire les id de l'utilisateur à prédire
      user_embedding = U_test[l_u] # extraire l'embedding du user actuel

      true_labels.append(extractNextUserState(batch))

      pred.append(model.predict_user_state(user_embedding))


  return true_labels,pred

true,pred = evaluate_classif(model, test, test_t_batch, U_test, I_test) 