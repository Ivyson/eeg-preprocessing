import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import mne
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mne.time_frequency import psd_array_welch
from scipy.signal import welch

pg.setConfigOptions(useOpenGL=False)
pg.setConfigOptions(antialias=True)  # smoother lines


class EEGVisualization(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        # EEG parameters
        self.n_channels = 8
        self.sample_rate = 250  # Hz
        self.window_size = 500  # samples to display
        self.buffer_size = self.window_size

        # Channel names
        self.channel_names = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]

        # Create MNE Info object with standard montage
        self.info = mne.create_info(
            ch_names=self.channel_names, sfreq=self.sample_rate, ch_types="eeg"
        )
        self.info.set_montage("standard_1020")

        # Data buffers (n_channels x buffer_size)
        self.signal_buffer = np.zeros((self.n_channels, self.buffer_size))
        self.time_axis = np.arange(self.buffer_size) / self.sample_rate

        # Stacking offsets (vertical offsets for each channel)
        self.offset_step = 60.0  # µV per channel offset; adjust visually
        # Offsets arranged top-to-bottom
        self.offsets = np.arange(self.n_channels)[::-1] * self.offset_step

        # Channel visibility flags
        self.channel_visible = [True] * self.n_channels

        # Classification -> single confidence value and rolling buffer
        self.confidence = 0.5
        self.conf_buffer = np.zeros(self.buffer_size)
        self.conf_time = np.arange(-self.buffer_size + 1, 1) / self.sample_rate

        # PSD parameters
        self.psd_nperseg = 256
        self.psd_freq_limit = 40.0  # Hz to show on PSD plot

        # Colors for channels
        self.colors = [
            pg.intColor(i, hues=self.n_channels, values=200, maxValue=255)
            for i in range(self.n_channels)
        ]

        # Setup UI
        self.setup_ui()

        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(40)  # ~25 FPS

        # Simulation time
        self.t = 0

    def setup_ui(self):
        self.setWindowTitle("Real-Time EEG Visualization (stacked traces + PSD)")
        self.setGeometry(100, 100, 1600, 1000)

        main_layout = QtWidgets.QVBoxLayout()

        # Top: signal plot (left) and PSD + collapsible legend (right)
        top_hlayout = QtWidgets.QHBoxLayout()

        # Single consolidated stacked signal plot (all channels overlayed with offsets)
        self.signal_plot = pg.PlotWidget(title="EEG Signals (stacked traces)")
        self.signal_plot.setXRange(0, self.window_size / self.sample_rate)
        self.signal_plot.showGrid(x=True, y=True, alpha=0.3)
        self.signal_plot.setLabel("bottom", "Time", units="s")
        self.signal_plot.setLabel("left", "Amplitude + offsets (µV)")
        # Y-range to show stacked traces
        total_height = self.offset_step * (self.n_channels - 1) + 200
        self.signal_plot.setYRange(-100, total_height - 100)

        # Create one curve per channel, overlayed in the same axes (with offsets)
        self.signal_curves = []
        for i in range(self.n_channels):
            pen = pg.mkPen(color=self.colors[i], width=1.5)
            curve = self.signal_plot.plot(
                self.time_axis,
                self.signal_buffer[i] + self.offsets[i],
                pen=pen,
                name=self.channel_names[i],
            )
            self.signal_curves.append(curve)

        # Right side panel: PSD plot + collapsible legend controls
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout()

        # PSD widget
        self.psd_widget = pg.PlotWidget(title="Power Spectral Density")
        self.psd_widget.setLabel("left", "PSD (dB/Hz)")
        self.psd_widget.setLabel("bottom", "Frequency (Hz)")
        self.psd_widget.showGrid(x=True, y=True, alpha=0.3)
        self.psd_widget.setXRange(0, self.psd_freq_limit)
        self.psd_widget.setYRange(-100, 20)
        self.psd_curves = []
        for i in range(self.n_channels):
            pen = pg.mkPen(color=self.colors[i], width=1, style=QtCore.Qt.PenStyle.DashLine)
            c = self.psd_widget.plot([], [], pen=pen, name=self.channel_names[i], alpha=0.5)
            self.psd_curves.append(c)
        # Avg PSD 
        self.psd_avg_curve = self.psd_widget.plot([], [], pen=pg.mkPen("k", width=2))

        # Legend Expander Buttons
        self.legend_toggle_btn = QtWidgets.QPushButton("Show Legend")
        self.legend_toggle_btn.setCheckable(True)
        self.legend_toggle_btn.toggled.connect(self.toggle_legend_panel)
        self.legend_panel = QtWidgets.QWidget()
        legend_layout = QtWidgets.QVBoxLayout()
        legend_layout.setContentsMargins(2, 2, 2, 2)
        self.checkboxes = []
        for i, name in enumerate(self.channel_names):
            h = QtWidgets.QHBoxLayout()
            color_label = QtWidgets.QLabel()
            color_label.setFixedSize(16, 16)

            color_label.setStyleSheet(f"background-color: {self.colors[i].name()};")
            cb = QtWidgets.QCheckBox(name)
            cb.setChecked(True)
            cb.toggled.connect(self.make_visibility_handler(i))
            self.checkboxes.append(cb)
            h.addWidget(color_label)
            h.addWidget(cb)
            h.addStretch()
            legend_layout.addLayout(h)
        legend_layout.addStretch()
        self.legend_panel.setLayout(legend_layout)
        self.legend_panel.setVisible(False)  # start collapsed

  
        right_layout.addWidget(self.psd_widget, stretch=3)
        right_layout.addWidget(self.legend_toggle_btn, stretch=0)
        right_layout.addWidget(self.legend_panel, stretch=1)
        right_panel.setLayout(right_layout)

        top_hlayout.addWidget(self.signal_plot, stretch=3)
        top_hlayout.addWidget(right_panel, stretch=1)

        bottom_widget = QtWidgets.QWidget()
        bottom_layout = QtWidgets.QHBoxLayout()

        self.headplot_canvas = MplCanvas(width=5, height=5, dpi=100)
        self.setup_headplot()

        # Confidence plot (0 - 100%)
        self.conf_widget = pg.PlotWidget(title="Concentration Percentage (%)")
        self.conf_widget.setBackground("w")
        self.conf_widget.showGrid(x=True, y=True, alpha=0.3)
        self.conf_widget.setYRange(0, 100)
        self.conf_widget.getAxis("left").setLabel("Percentage (%)")
        self.conf_curve = self.conf_widget.plot(self.conf_time, self.conf_buffer, pen=pg.mkPen("b", width=2))

        self.conf_label = QtWidgets.QLabel("0 %")
        font = self.conf_label.font()
        font.setPointSize(20)
        font.setBold(True)
        self.conf_label.setFont(font)
        self.conf_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        conf_layout_widget = QtWidgets.QWidget()
        conf_layout = QtWidgets.QVBoxLayout()
        conf_layout.addWidget(self.conf_widget, stretch=3)
        conf_layout.addWidget(self.conf_label, stretch=1)
        conf_layout_widget.setLayout(conf_layout)

        bottom_layout.addWidget(self.headplot_canvas, stretch=1)
        bottom_layout.addWidget(conf_layout_widget, stretch=1)
        bottom_widget.setLayout(bottom_layout)

        # Compose main layout
        main_layout.addLayout(top_hlayout, stretch=3)
        main_layout.addWidget(bottom_widget, stretch=1)

        self.setLayout(main_layout)

    def make_visibility_handler(self, idx):
        # return a closure that toggles channel idx visibility
        def handler(checked):
            self.channel_visible[idx] = checked
            self.signal_curves[idx].setVisible(checked)
            self.psd_curves[idx].setVisible(checked)

        return handler

    def toggle_legend_panel(self, checked):
        self.legend_panel.setVisible(checked)
        self.legend_toggle_btn.setText("Hide Legend" if checked else "Show Legend")

    def setup_headplot(self):
        """Initial headplot setup if any (WILL UPDATE LATER ON!)."""
        pass

    def update_headplot(self, data):
        """Update the topographic map with new data using MNE (8 channels)."""
        self.headplot_canvas.figure.clear()
        ax = self.headplot_canvas.figure.add_subplot(111)

        mne.viz.plot_topomap(
            data,
            self.info,
            axes=ax,
            show=False,
            contours=10,
            cmap="RdBu_r",
            sensors=False,
        )

        ax.set_title("Scalp (µV)", fontsize=12, fontweight="bold")
        self.headplot_canvas.figure.tight_layout()
        self.headplot_canvas.draw()

    def generate_fake_eeg(self):
        """Generate realistic fake EEG data (one sample per channel)."""
        alpha = 10  # Hz
        beta = 20
        theta = 6

        new_sample = np.zeros(self.n_channels)
        for i in range(self.n_channels):
            signal = (
                20 * np.sin(2 * np.pi * alpha * self.t + i * 0.5)
                + 10 * np.sin(2 * np.pi * beta * self.t + i * 0.3)
                + 15 * np.sin(2 * np.pi * theta * self.t + i * 0.8)
                + np.random.randn() * 5
            )

            if (self.t % 2) < 1:
                if "F3" in self.channel_names[i] or "C3" in self.channel_names[i]:
                    signal += 30 * np.sin(2 * np.pi * 12 * self.t)
            else:
                if "F4" in self.channel_names[i] or "C4" in self.channel_names[i]:
                    signal += 30 * np.sin(2 * np.pi * 12 * self.t)

            new_sample[i] = signal

        self.t += 1 / self.sample_rate
        return new_sample

    def update_classification(self):
        """Simulates a single confidence value and update rolling buffer."""
        base = np.clip(np.random.normal(0.6, 0.15), 0.0, 1.0)
        self.confidence = 0.9 * self.confidence + 0.1 * base
        self.conf_buffer = np.roll(self.conf_buffer, -1)
        self.conf_buffer[-1] = self.confidence * 100.0



    def compute_and_update_psd(self):
        """Compute PSD for each channel from rolling buffer using MNE."""
        info  = mne.create_info(ch_names=self.channel_names, sfreq=self.sample_rate, ch_types="eeg")
        raw = mne.io.RawArray(self.signal_buffer, info)
        raw.filter(fmin = 0.5,fmax= self.psd_freq_limit, fir_design='firwin', verbose=False)
        psds = raw.compute_psd(method='welch', n_fft=self.psd_nperseg*2, n_overlap=1, n_per_seg=self.psd_nperseg, fmin=0.5, fmax=self.psd_freq_limit, verbose=False)
        fmin, fmax = 0.5, self.psd_freq_limit
        n_per_seg = min(self.psd_nperseg, self.buffer_size)
        # returns psds (n_channels, n_freqs) and freqs (n_freqs,)
        psds, freqs = psd_array_welch(self.signal_buffer, sfreq=self.sample_rate,
                                    fmin=fmin, fmax=fmax, n_fft=n_per_seg*2,
                                    n_overlap=0, n_per_seg=n_per_seg, average='mean',
                                    verbose=False)
        # psds are in V^2/Hz if your data is in volts — convert to dB
        psds_db = 10.0 * np.log10(psds + 1e-12)
        # update per-channel curves
        for i in range(self.n_channels):
            self.psd_curves[i].setData(freqs, psds_db[i])

        avg_db = np.mean(psds_db, axis=0)
        self.psd_avg_curve.setData(freqs, avg_db)
    def update_plots(self):
        """Update signal overlay, headplot and confidence trace."""
        new_sample = self.generate_fake_eeg()

        # shift stack and add teh new sampled data..
        self.signal_buffer = np.roll(self.signal_buffer, -1, axis=1)
        self.signal_buffer[:, -1] = new_sample

        for i, curve in enumerate(self.signal_curves):
            y = self.signal_buffer[i] + self.offsets[i]
            curve.setData(self.time_axis, y)
            curve.setVisible(self.channel_visible[i])

        # update headplot with instantaneous values
        self.update_headplot(new_sample)

        # update confidence
        self.update_classification()
        self.conf_curve.setData(self.conf_time, self.conf_buffer)
        self.conf_label.setText(f"{self.conf_buffer[-1]:.1f} %")

        # update PSD (from rolling buffers)
        self.compute_and_update_psd()


class MplCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for embedding in Qt"""
    def __init__(self, parent=None, width=10, height=15, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = EEGVisualization()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()