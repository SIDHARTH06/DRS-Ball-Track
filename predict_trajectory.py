import numpy as np

def predict_trajectory(ball_detections,predicted_trajectory=[]):
    # Define parameters for Kalman filter
    dt = 1.0/30.0 # Time step
    # State transition matrix
    A = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
    # Measurement matrix
    H = np.array([[1, 0, 0], [0, 1, 0]])
    # Process noise covariance
    Q = np.diag([0.1, 0.1, 0.1])
    # Measurement noise covariance
    R = np.diag([0.1, 0.1])

    # Initialize state and covariance matrices
    x = np.zeros((3, 1))
    P = np.diag([1000, 1000, 1000])

    # Define arrays to store predicted trajectory and ball positions
    if ball_detections is None:
        pass
    else:
        for detection in ball_detections:
        # Get current ball position
  
        # Predict next state using Kalman filter
            x_pred = np.dot(A, x)
            P_pred = np.dot(A, np.dot(P, A.T)) + Q

        # Update state using measurement
            z = np.array(detection).reshape((2, 1))
            y = z - np.dot(H, x_pred)
            S = np.dot(H, np.dot(P_pred, H.T)) + R
            K = np.dot(P_pred, np.dot(H.T, np.linalg.inv(S)))
            x = x_pred + np.dot(K, y)
            P = np.dot((np.eye(3) - np.dot(K, H)), P_pred)

        # Store predicted position
            x_pred = np.round(x_pred).astype(int)
            predicted_position = (x_pred[0][0], x_pred[1][0])
        predicted_trajectory.append(predicted_position)

    return predicted_trajectory
