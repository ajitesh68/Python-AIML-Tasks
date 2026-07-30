import tensorflow as tf 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense 


x = tf.constant([[1.0], [2.0], [3.0], [4.0]])
y = 2*x + 1 

def train_model(loss_fn,optimizer_name, epochs=200):
    model = Sequential([Dense(1,input_shape=(1,))])
    if loss_fn == 'mse':
        loss = 'mse'
    elif loss_fn == 'mae':
        loss = 'mae'
    
    if optimizer_name == 'sgd':
        opt = 'sgd'
    elif optimizer_name == 'adam':
        opt = 'adam'

    model.compile(optimizer=opt, loss=loss)
    history = model.fit(x,y,epochs=epochs,verbose=0)  
    final_loss = history.history['loss'][-1]  

    weights = model.layers[0].get_weights()
    w, b = weights[0][0][0], weights[1][0]
    return final_loss,w,b

# Run combinations
combinations = [('mse','sgd'), ('mse','adam'), ('mae','sgd'), ('mae','adam')]
results = {}
for loss_fn, opt_name in combinations:
    print(f"\n--- Training with {loss_fn.upper()} + {opt_name.upper()} ---")
    final_loss, w, b = train_model(loss_fn, opt_name, epochs=200)  # store full losses list for plotting
    results[(loss_fn, opt_name)] = (final_loss, w, b)

# 4. Summary
print("\n=== SUMMARY ===")
for (loss_fn, opt_name), (final_loss, w, b) in results.items():
    print(f"{loss_fn.upper():3} + {opt_name.upper():4} | Final Loss: {final_loss:.6f} | w={w:.4f}, b={b:.4f}")