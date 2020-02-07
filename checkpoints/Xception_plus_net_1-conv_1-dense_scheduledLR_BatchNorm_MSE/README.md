# network with Xception as input plus 1 convolutional layers (to 64) with BatchNormalization and 1 dense (37 outputs), LR with scheduler (starting from 1e-3), Adam optimizer and 'MSE' as loss function

We tried Xception plus some layers in this way (22 million parameters more or less), obtaining these following results:
## RMSE = 0.07971, Validation Loss = 0.00628

_here is the NN used:_

	def CNN_galaxy(model):
	  
		inp = model    
		model = Sequential()
		model.add(inp)
		model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(64, (3, 3)))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		
		model.add(GlobalMaxPooling2D())

		#model.add(Flatten())		
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
