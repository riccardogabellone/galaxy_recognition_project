# network with Xception as input plus 1 dense (37 outputs), LR with scheduler (starting from 1e-3), Adam optimizer and 'MSE' as loss function

We split training dataset into [test](test_from_training/) set (20%) and [train](training_from_training/) set (80%), using it 10% for validation and 90% for training, obtaining these following results:
## RMSE on train = 0.0744, RMSE on test = 0.07298, Validation Loss = 0.00598

_here is the NN used:_

	def CNN_galaxy(model):
	  
		inp = model    
		model = Sequential()
		model.add(inp)

		model.add(Flatten())		
		model.add(Dropout(0.25))
		model.add(Dense(NUM_CLASSES))
		model.add(Activation('sigmoid'))

		for layer in model.layers:
		  layer.trainable = True

		print("compiling the model...")
		optimizer = tf.keras.optimizers.Adam(lr=LR, decay=WEIGHT_DECAY)
		model.compile(optimizer, loss='mse', metrics=[root_mean_squared_error])

		return model


	img_shape = (224, 224, 3)
	xception_model = tf.keras.applications.xception.Xception(include_top=False, input_shape=img_shape)
	net = CNN_galaxy(xception_model)
