def do_gradient_descent():
    w, b, eta = inti_w, init_b, 1.0
    prev__v_w, prev_v_b, gamma = 0, 0, 0.9
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y)
            dw += grad_w(w, b, x, y)
            db += grad_w(w, b, x, y)
        v_w = gamma * prev_v_w + eta * dw
        v_b = gamma * prev_v_b + eta * db
        w = w - v_w
        b = b - v_b