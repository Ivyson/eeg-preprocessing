import serial
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# Use the model trained
model = load("models/attention_dataset.joblib")
scaler = StandardScaler()

# tHIS FOR STORING DATA
max_points = 50
delta_data = deque(maxlen=max_points)
theta_data = deque(maxlen=max_points)
lowAlpha_data = deque(maxlen=max_points)
highAlpha_data = deque(maxlen=max_points)
lowBeta_data = deque(maxlen=max_points)
highBeta_data = deque(maxlen=max_points)
lowGamma_data = deque(maxlen=max_points)
highGamma_data = deque(maxlen=max_points)
predictions = deque(maxlen=max_points)
timestamps = deque(maxlen=max_points)

time_counter = 0
latest_data = None


def parse_packet(packet):
    if len(packet) < 4 or packet[0] != 0xAA or packet[1] != 0xAA:
        return None

    payload_len = packet[2]

    if payload_len == 32 and packet[3] == 0x02 and packet[5] == 0x83:
        i = 7
        delta = (packet[i] << 16) | (packet[i + 1] << 8) | packet[i + 2]
        theta = (packet[i + 3] << 16) | (packet[i + 4] << 8) | packet[i + 5]
        lowAlpha = (packet[i + 6] << 16) | (packet[i + 7] << 8) | packet[i + 8]
        highAlpha = (packet[i + 9] << 16) | (packet[i + 10] << 8) | packet[i + 11]
        lowBeta = (packet[i + 12] << 16) | (packet[i + 13] << 8) | packet[i + 14]
        highBeta = (packet[i + 15] << 16) | (packet[i + 16] << 8) | packet[i + 17]
        lowGamma = (packet[i + 18] << 16) | (packet[i + 19] << 8) | packet[i + 20]
        highGamma = (packet[i + 21] << 16) | (packet[i + 22] << 8) | packet[i + 23]

        attention = packet[31]
        meditation = packet[33]
        quality = packet[4]

        result = 1 if attention >= 50 else 0
        features = [
            attention,
            meditation,
            delta,
            theta,
            lowAlpha,
            highAlpha,
            lowBeta,
            highBeta,
            lowGamma,
            highGamma,
            result,
        ]

        return {
            "features": features,
            "delta": delta,
            "theta": theta,
            "lowAlpha": lowAlpha,
            "highAlpha": highAlpha,
            "lowBeta": lowBeta,
            "highBeta": highBeta,
            "lowGamma": lowGamma,
            "highGamma": highGamma,
            "attention": attention,
            "meditation": meditation,
            "quality": quality,
        }
    return None


# Serial Comm, This works for my MAC, For Windows you would use COM3 or COM4 etc
port = "/dev/tty.usbmodem2017_2_251"
baud_rate = 57600
ser = serial.Serial(port, baud_rate, timeout=1)
buffer = []

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Brain Data Plot", fontsize=16, fontweight="bold")

ax1.set_title("Frequency Bands")
ax1.set_ylabel("Power")
ax1.set_ylim(0, 100)
ax1.grid(True, alpha=0.3)
(line_delta,) = ax1.plot([], [], "-", linewidth=2, label="Delta")
(line_theta,) = ax1.plot([], [], "-", linewidth=2, label="Theta")
(line_lowAlpha,) = ax1.plot([], [], "-", linewidth=2, label="Low Alpha")
(line_highAlpha,) = ax1.plot([], [], "-", linewidth=2, label="High Alpha")
(line_lowBeta,) = ax1.plot([], [], "-", linewidth=2, label="Low Beta")
(line_highBeta,) = ax1.plot([], [], "-", linewidth=2, label="High Beta")
(line_lowGamma,) = ax1.plot([], [], "-", linewidth=2, label="Low Gamma")
(line_highGamma,) = ax1.plot([], [], "-", linewidth=2, label="High Gamma")
ax1.legend(loc="upper right", ncol=2)

ax2.set_title("Model Prediction")
ax2.set_ylabel("Classification")
ax2.set_xlabel("Sample")
ax2.set_ylim(-0.5, 1.5)
ax2.set_yticks([0, 1])
ax2.grid(True, alpha=0.3)
(line_pred,) = ax2.plot([], [], "b-", linewidth=2, marker="o", markersize=6)

plt.tight_layout()


def update(frame):
    global buffer, time_counter, latest_data

    # Read serial data
    if ser.in_waiting > 0:
        new_bytes = ser.read(ser.in_waiting)

        for byte in new_bytes:
            value = byte if isinstance(byte, int) else ord(byte)
            buffer.append(value)

            if len(buffer) >= 2 and buffer[-2:] == [0xAA, 0xAA]:
                if len(buffer) > 2:
                    packet = buffer[:-2]
                    if packet and packet[0] == 0xAA:
                        data = parse_packet(packet)
                        if data:
                            # Predict
                            X = np.array([data["features"]]).reshape(1, -1)
                            X_scaled = scaler.fit_transform(X)
                            pred = model.predict(X_scaled)[0]

                            # Store raw data (no normalisation yet)...
                            delta_data.append(data["delta"])
                            theta_data.append(data["theta"])
                            lowAlpha_data.append(data["lowAlpha"])
                            highAlpha_data.append(data["highAlpha"])
                            lowBeta_data.append(data["lowBeta"])
                            highBeta_data.append(data["highBeta"])
                            lowGamma_data.append(data["lowGamma"])
                            highGamma_data.append(data["highGamma"])
                            predictions.append(int(pred))
                            timestamps.append(time_counter)
                            time_counter += 1

                            latest_data = data

                buffer = [0xAA, 0xAA]

    # Update THE plots...
    if len(timestamps) > 0:
        line_delta.set_data(list(timestamps), list(delta_data))
        line_theta.set_data(list(timestamps), list(theta_data))
        line_lowAlpha.set_data(list(timestamps), list(lowAlpha_data))
        line_highAlpha.set_data(list(timestamps), list(highAlpha_data))
        line_lowBeta.set_data(list(timestamps), list(lowBeta_data))
        line_highBeta.set_data(list(timestamps), list(highBeta_data))
        line_lowGamma.set_data(list(timestamps), list(lowGamma_data))
        line_highGamma.set_data(list(timestamps), list(highGamma_data))
        line_pred.set_data(list(timestamps), list(predictions))

        ax1.set_xlim(max(0, time_counter - max_points), time_counter + 1)
        ax2.set_xlim(max(0, time_counter - max_points), time_counter + 1)

    if latest_data:
        pred_text = predictions[-1] if len(predictions) > 0 else "N/A"
        fig.suptitle(
            f"Real-time data | Prediction: {pred_text} | Quality: {latest_data['quality']}",
            fontsize=16,
            fontweight="bold",
        )

    return (
        line_delta,
        line_theta,
        line_lowAlpha,
        line_highAlpha,
        line_lowBeta,
        line_highBeta,
        line_lowGamma,
        line_highGamma,
        line_pred,
    )


ani = animation.FuncAnimation(
    fig, update, interval=50, blit=True, cache_frame_data=False
)

print("Starting live prediction...")
plt.show()

ser.close()
