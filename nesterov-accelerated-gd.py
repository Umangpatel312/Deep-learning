def do_nesterov_accelerated_gradient_descent():
    w,b,eta=init_w,init_b=1.0
    prev_v_w,prev_v_b=0,0,0.9
    for i in range(max_epochs):
        dw,db,=0,0
        #do partial updates
        v_w=gamma*prev_v_w
        v_b=gamma*prev_v_b
        for x,y in zip(X,Y)
            #calculate gradients after partial update
            dw+=grad_w(w-v_w,b-)