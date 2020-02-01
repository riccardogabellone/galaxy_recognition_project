# network with 5 convolutional layers (from 512 to 32) and 3*3+1 dense, LR = 1e-3 and Adamax optimizer

## RMSE = 0.1339, Validation Loss = 0.2783

_here is the NN used:_

	def CNN_galaxy():

	    model = Sequential()
	    model.add(Conv2D(512, (3, 3), input_shape=(224, 224, 3)))
	    model.add(Conv2D(256, (3, 3)))
	    #model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(MaxPooling2D(pool_size=(2, 2)))

	    model.add(Conv2D(256, (3, 3)))
	    model.add(Conv2D(128, (3, 3)))
	    #model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(MaxPooling2D(pool_size=(2, 2)))

	    model.add(Conv2D(128, (3, 3)))
	    model.add(Conv2D(64, (3, 3)))
	    #model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(MaxPooling2D(pool_size=(2, 2)))

	    model.add(Conv2D(64, (3, 3)))
	    model.add(Conv2D(32, (3, 3)))
	    #model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(MaxPooling2D(pool_size=(2, 2)))

	    model.add(Conv2D(32, (3, 3)))
	    model.add(Conv2D(32, (3, 3)))
	    #model.add(BatchNormalization())
	    model.add(Activation('relu'))
	    model.add(GlobalMaxPooling2D())


	    model.add(Dropout(0.25))
	    model.add(Dense(32))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.20)) 
	    model.add(Dense(32))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.15)) #0.25
	    model.add(Dense(32))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.10)) #0.25


	    model.add(Dropout(0.25))
	    model.add(Dense(32))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.20)) 
	    model.add(Dense(32))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.15)) #0.25
	    model.add(Dense(32))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.10)) #0.25


	    model.add(Dropout(0.25))
	    model.add(Dense(32))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.20)) 
	    model.add(Dense(32))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.15)) #0.25
	    model.add(Dense(32))
	    model.add(Activation('relu'))
	    model.add(Dropout(0.10)) #0.25


	    model.add(Dense(NUM_CLASSES))
	    model.add(Activation('sigmoid'))

	    for layer in model.layers:
	      layer.trainable = True

	    LR = 1e-3
	    optimizer = keras.optimizers.Adamax(lr=LR, decay=WEIGHT_DECAY)
	    model.compile(optimizer, loss='binary_crossentropy', metrics=[root_mean_squared_error])

	    return model
