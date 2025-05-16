import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageEnhance
import cv2
import numpy as np
from matplotlib import pyplot as plt

BUTTON_COLOR = "#1D2545"
BG_COLOR = "#000435"
BUTTON_FONT = ("Arial", 9, "bold")

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Görüntü İşleme Uygulaması")
        self.root.configure(bg=BG_COLOR)
        self.image = None
        self.processed_image = None
        self.cv_image = None
        self.cv_processed = None
        self.original_image = None
        self.zoom_factor = 1.0
        self.create_widgets()

    def create_widgets(self):
        left_frame = tk.Frame(self.root, bg=BG_COLOR)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        right_frame = tk.Frame(self.root, bg=BG_COLOR)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.buttons = [
            ("Görsel Ekle", self.load_image),
            ("Görsel Kaydet", self.save_image),
            ("Orijinaline Çevir", self.to_original),
            ("Griye Çevir", self.to_gray),
            ("Negatif", self.negative),
            ("Grayscale", self.to_gray),
            ("Parlaklık +", lambda: self.adjust_brightness(1.2)),
            ("Parlaklık -", lambda: self.adjust_brightness(0.8)),
            ("Eşikleme", self.threshold),
            ("Histogram", self.show_histogram),
            ("Hist. Eşitle", self.hist_equalize),
            ("Kontrast +", lambda: self.adjust_contrast(1.2)),
            ("Kontrast -", lambda: self.adjust_contrast(0.8)),
            ("Taşı", self.translate),
            ("Aynalama", self.mirror),
            ("Eğme", self.shear),
            ("Döndür", self.rotate),
            ("Kırp", self.crop),
            ("Perspektif", self.perspective_transform),
            ("Ortalama Filtre", self.mean_filter),
            ("Medyan Filtre", self.median_filter),
            ("Gauss Filtre", self.gaussian_filter),
            ("Konservatif Filtre", self.conservative_filter),
            ("Crimmins Speckle", self.crimmins_speckle),
            ("Fourier LPF", lambda: self.fourier_filter("lpf")),
            ("Fourier HPF", lambda: self.fourier_filter("hpf")),
            ("Band Geçiren", lambda: self.fourier_filter("bandpass")),
            ("Band Durduran", lambda: self.fourier_filter("bandstop")),
            ("Butterworth LPF", lambda: self.butterworth_filter("lpf")),
            ("Butterworth HPF", lambda: self.butterworth_filter("hpf")),
            ("Gaussian LPF", lambda: self.gaussian_freq_filter("lpf")),
            ("Gaussian HPF", lambda: self.gaussian_freq_filter("hpf")),
            ("Homomorfik", self.homomorphic_filter),
            ("Sobel", self.sobel),
            ("Prewitt", self.prewitt),
            ("Roberts Cross", self.roberts_cross),
            ("Compass", self.compass),
            ("Canny", self.canny),
            ("Laplace", self.laplace),
            ("Gabor", self.gabor),
            ("Hough", self.hough_transform),
            ("K-means", self.kmeans_segmentation),
            ("Erode", self.erode),
            ("Dilate", self.dilate),
        ]

        for i, (text, cmd) in enumerate(self.buttons):
            btn = tk.Button(
                left_frame, text=text, command=cmd, bg=BUTTON_COLOR, fg="white",
                font=BUTTON_FONT, width=16, height=2
            )
            btn.grid(row=i // 4, column=i % 4, padx=2, pady=2, sticky="nsew")

        self.original_label = tk.Label(right_frame, text="Orijinal Görsel", bg=BG_COLOR, fg="white")
        self.original_label.pack(pady=5)
        self.original_canvas = tk.Label(right_frame, bg=BG_COLOR)
        self.original_canvas.pack(pady=5)

        self.processed_label = tk.Label(right_frame, text="İşlenmiş Görsel", bg=BG_COLOR, fg="white")
        self.processed_label.pack(pady=5)
        self.processed_canvas = tk.Label(right_frame, bg=BG_COLOR)
        self.processed_canvas.pack(pady=5)

        # Küçük RGB scroll barlar sağda, görsellerin altında
        rgb_frame = tk.LabelFrame(right_frame, text="RGB Ayarları", bg=BG_COLOR, fg="white", font=("Arial", 8, "bold"))
        rgb_frame.pack(fill="x", padx=5, pady=8)

        self.r_scale = tk.Scale(
            rgb_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Kırmızı",
            bg=BG_COLOR, fg="red", troughcolor=BUTTON_COLOR, command=self.update_rgb,
            length=120, sliderlength=10, width=7, font=("Arial", 7)
        )
        self.r_scale.set(128)
        self.r_scale.pack(fill="x", padx=3, pady=0)

        self.g_scale = tk.Scale(
            rgb_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Yeşil",
            bg=BG_COLOR, fg="green", troughcolor=BUTTON_COLOR, command=self.update_rgb,
            length=120, sliderlength=10, width=7, font=("Arial", 7)
        )
        self.g_scale.set(128)
        self.g_scale.pack(fill="x", padx=3, pady=0)

        self.b_scale = tk.Scale(
            rgb_frame, from_=0, to=255, orient=tk.HORIZONTAL, label="Mavi",
            bg=BG_COLOR, fg="blue", troughcolor=BUTTON_COLOR, command=self.update_rgb,
            length=120, sliderlength=10, width=7, font=("Arial", 7)
        )
        self.b_scale.set(128)
        self.b_scale.pack(fill="x", padx=3, pady=0)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image = Image.open(file_path).convert("RGB")
            self.original_image = self.image.copy()
            self.cv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2BGR)
            self.processed_image = self.image.copy()
            self.cv_processed = self.cv_image.copy()
            self.zoom_factor = 1.0
            self.r_scale.set(128)
            self.g_scale.set(128)
            self.b_scale.set(128)
            self.display_images()

    def to_original(self):
        if self.original_image:
            self.image = self.original_image.copy()
            self.processed_image = self.original_image.copy()
            self.cv_image = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
            self.cv_processed = self.cv_image.copy()
            self.zoom_factor = 1.0
            self.r_scale.set(128)
            self.g_scale.set(128)
            self.b_scale.set(128)
            self.display_images()

    def zoom_in(self):
        if self.original_image:
            self.zoom_factor *= 1.2
            self.apply_zoom()

    def zoom_out(self):
        if self.original_image:
            self.zoom_factor /= 1.2
            self.apply_zoom()

    def apply_zoom(self):
        if self.original_image:
            w, h = self.original_image.size
            new_size = (max(1, int(w * self.zoom_factor)), max(1, int(h * self.zoom_factor)))
            zoomed = self.original_image.resize(new_size, Image.LANCZOS)
            self.image = zoomed
            self.processed_image = zoomed.copy()
            self.display_images()

    def save_image(self):
        if self.processed_image:
            file_path = filedialog.asksaveasfilename(defaultextension=".png")
            if file_path:
                self.processed_image.save(file_path)
                messagebox.showinfo("Başarılı", "Görsel kaydedildi.")
        else:
            messagebox.showwarning("Uyarı", "Önce bir görsel yükleyin ve işleyin.")

    def display_images(self):
        if self.image:
            img = self.image.resize((256, 256))
            imgtk = ImageTk.PhotoImage(img)
            self.original_canvas.imgtk = imgtk
            self.original_canvas.config(image=imgtk)
        if self.processed_image:
            img = self.processed_image.resize((256, 256))
            imgtk = ImageTk.PhotoImage(img)
            self.processed_canvas.imgtk = imgtk
            self.processed_canvas.config(image=imgtk)

    def to_gray(self):
        if self.image:
            gray = self.processed_image.convert("L")
            self.processed_image = gray.convert("RGB")
            self.cv_processed = cv2.cvtColor(np.array(self.processed_image), cv2.COLOR_RGB2BGR)
            self.display_images()

    def update_rgb(self, event=None):
        if self.image:
            r = self.r_scale.get()
            g = self.g_scale.get()
            b = self.b_scale.get()
            arr = np.array(self.image).astype(np.float32)
            arr[..., 0] = np.clip(arr[..., 0] * (r / 128), 0, 255)
            arr[..., 1] = np.clip(arr[..., 1] * (g / 128), 0, 255)
            arr[..., 2] = np.clip(arr[..., 2] * (b / 128), 0, 255)
            self.processed_image = Image.fromarray(arr.astype(np.uint8))
            self.cv_processed = cv2.cvtColor(np.array(self.processed_image), cv2.COLOR_RGB2BGR)
            self.display_images()

    def negative(self):
        if self.image:
            arr = np.array(self.processed_image)
            arr = 255 - arr
            self.processed_image = Image.fromarray(arr)
            self.cv_processed = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            self.display_images()

    def adjust_brightness(self, factor):
        if self.image:
            enhancer = ImageEnhance.Brightness(self.processed_image)
            self.processed_image = enhancer.enhance(factor)
            self.cv_processed = cv2.cvtColor(np.array(self.processed_image), cv2.COLOR_RGB2BGR)
            self.display_images()

    def threshold(self):
        if self.image:
            thresh = simpledialog.askinteger("Eşikleme", "Eşik değeri (0-255):", minvalue=0, maxvalue=255)
            gray = self.processed_image.convert("L")
            arr = np.array(gray)
            arr = np.where(arr > thresh, 255, 0).astype(np.uint8)
            self.processed_image = Image.fromarray(arr).convert("RGB")
            self.cv_processed = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            self.display_images()

    def show_histogram(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            plt.figure("Histogram")
            plt.hist(arr.flatten(), bins=256, range=[0,256], color='gray')
            plt.title("Histogram")
            plt.xlabel("Piksel Değeri")
            plt.ylabel("Frekans")
            plt.show()

    def hist_equalize(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            eq = cv2.equalizeHist(arr)
            self.processed_image = Image.fromarray(eq).convert("RGB")
            self.cv_processed = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
            self.display_images()

    def adjust_contrast(self, factor):
        if self.image:
            enhancer = ImageEnhance.Contrast(self.processed_image)
            self.processed_image = enhancer.enhance(factor)
            self.cv_processed = cv2.cvtColor(np.array(self.processed_image), cv2.COLOR_RGB2BGR)
            self.display_images()

    def translate(self):
        if self.image:
            dx = simpledialog.askinteger("Taşı", "X ekseni (piksel):", initialvalue=20)
            dy = simpledialog.askinteger("Taşı", "Y ekseni (piksel):", initialvalue=20)
            arr = np.array(self.processed_image)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(arr, M, (arr.shape[1], arr.shape[0]))
            self.processed_image = Image.fromarray(shifted)
            self.cv_processed = cv2.cvtColor(shifted, cv2.COLOR_RGB2BGR)
            self.display_images()

    def mirror(self):
        if self.image:
            arr = np.array(self.processed_image)
            mirrored = np.fliplr(arr)
            self.processed_image = Image.fromarray(mirrored)
            self.cv_processed = cv2.cvtColor(mirrored, cv2.COLOR_RGB2BGR)
            self.display_images()

    def shear(self):
        if self.image:
            sh = simpledialog.askfloat("Eğme", "Shear katsayısı:", initialvalue=0.2)
            arr = np.array(self.processed_image)
            rows, cols, ch = arr.shape
            M = np.float32([[1, sh, 0], [0, 1, 0]])
            sheared = cv2.warpAffine(arr, M, (int(cols + sh * rows), rows))
            self.processed_image = Image.fromarray(sheared)
            self.cv_processed = cv2.cvtColor(sheared, cv2.COLOR_RGB2BGR)
            self.display_images()

    def rotate(self):
        if self.image:
            angle = simpledialog.askfloat("Döndür", "Açı (derece):", initialvalue=45)
            arr = np.array(self.processed_image)
            rows, cols = arr.shape[:2]
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(arr, M, (cols, rows))
            self.processed_image = Image.fromarray(rotated)
            self.cv_processed = cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR)
            self.display_images()

    def crop(self):
        if self.processed_image:
            width, height = self.processed_image.size
            x1 = simpledialog.askinteger("Kırp", "Başlangıç X:", minvalue=0, maxvalue=width-2)
            y1 = simpledialog.askinteger("Kırp", "Başlangıç Y:", minvalue=0, maxvalue=height-2)
            x2 = simpledialog.askinteger("Kırp", "Bitiş X:", minvalue=x1+1, maxvalue=width)
            y2 = simpledialog.askinteger("Kırp", "Bitiş Y:", minvalue=y1+1, maxvalue=height)
            if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
                cropped = self.processed_image.crop((x1, y1, x2, y2))
                self.processed_image = cropped
                self.cv_processed = cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)
                self.display_images()

    def perspective_transform(self):
        if self.image:
            arr = np.array(self.processed_image)
            rows, cols = arr.shape[:2]
            pts1 = np.float32([[0,0],[cols-1,0],[0,rows-1],[cols-1,rows-1]])
            pts2 = np.float32([
                [simpledialog.askinteger("Perspektif", "Sol üst X:", initialvalue=0), simpledialog.askinteger("Perspektif", "Sol üst Y:", initialvalue=0)],
                [simpledialog.askinteger("Perspektif", "Sağ üst X:", initialvalue=cols-1), simpledialog.askinteger("Perspektif", "Sağ üst Y:", initialvalue=0)],
                [simpledialog.askinteger("Perspektif", "Sol alt X:", initialvalue=0), simpledialog.askinteger("Perspektif", "Sol alt Y:", initialvalue=rows-1)],
                [simpledialog.askinteger("Perspektif", "Sağ alt X:", initialvalue=cols-1), simpledialog.askinteger("Perspektif", "Sağ alt Y:", initialvalue=rows-1)],
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            dst = cv2.warpPerspective(arr, M, (cols, rows))
            self.processed_image = Image.fromarray(dst)
            self.cv_processed = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
            self.display_images()

    def mean_filter(self):
        if self.image:
            arr = np.array(self.processed_image)
            mean = cv2.blur(arr, (3,3))
            self.processed_image = Image.fromarray(mean)
            self.cv_processed = cv2.cvtColor(mean, cv2.COLOR_RGB2BGR)
            self.display_images()

    def median_filter(self):
        if self.image:
            arr = np.array(self.processed_image)
            median = cv2.medianBlur(arr, 3)
            self.processed_image = Image.fromarray(median)
            self.cv_processed = cv2.cvtColor(median, cv2.COLOR_RGB2BGR)
            self.display_images()

    def gaussian_filter(self):
        if self.image:
            arr = np.array(self.processed_image)
            gauss = cv2.GaussianBlur(arr, (5,5), 0)
            self.processed_image = Image.fromarray(gauss)
            self.cv_processed = cv2.cvtColor(gauss, cv2.COLOR_RGB2BGR)
            self.display_images()

    def conservative_filter(self):
        if self.image:
            arr = np.array(self.processed_image)
            def conservative(arr):
                result = arr.copy()
                for i in range(1, arr.shape[0]-1):
                    for j in range(1, arr.shape[1]-1):
                        for k in range(arr.shape[2]):
                            local = arr[i-1:i+2, j-1:j+2, k].flatten()
                            min_val = local.min()
                            max_val = local.max()
                            if arr[i,j,k] < min_val:
                                result[i,j,k] = min_val
                            elif arr[i,j,k] > max_val:
                                result[i,j,k] = max_val
                return result
            filtered = conservative(arr)
            self.processed_image = Image.fromarray(filtered)
            self.cv_processed = cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR)
            self.display_images()

    def crimmins_speckle(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            def crimmins(arr):
                arr = arr.astype(np.int32)
                for _ in range(2):
                    for i in range(1, arr.shape[0]-1):
                        for j in range(1, arr.shape[1]-1):
                            n = arr[i-1:i+2, j-1:j+2].flatten()
                            arr[i,j] = np.median(n)
                return arr.astype(np.uint8)
            filtered = crimmins(arr)
            self.processed_image = Image.fromarray(filtered).convert("RGB")
            self.cv_processed = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            self.display_images()

    def fourier_filter(self, mode):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            dft = cv2.dft(np.float32(arr), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            mask = np.zeros((rows, cols, 2), np.uint8)
            r = 30
            if mode == "lpf":
                mask[crow-r:crow+r, ccol-r:ccol+r] = 1
            elif mode == "hpf":
                mask[:,:] = 1
                mask[crow-r:crow+r, ccol-r:ccol+r] = 0
            elif mode == "bandpass":
                mask[crow-40:crow+40, ccol-40:ccol+40] = 1
                mask[crow-20:crow+20, ccol-20:ccol+20] = 0
            elif mode == "bandstop":
                mask[:,:] = 1
                mask[crow-40:crow+40, ccol-40:ccol+40] = 0
                mask[crow-20:crow+20, ccol-20:ccol+20] = 1
            fshift = dft_shift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            self.processed_image = Image.fromarray(img_back.astype(np.uint8)).convert("RGB")
            self.cv_processed = cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.display_images()

    def butterworth_filter(self, mode):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            u = np.arange(rows)
            v = np.arange(cols)
            U, V = np.meshgrid(u, v, sparse=False, indexing='ij')
            D = np.sqrt((U-crow)**2 + (V-ccol)**2)
            n = 2
            D0 = 30
            if mode == "lpf":
                H = 1 / (1 + (D/D0)**(2*n))
            else:
                H = 1 / (1 + (D0/D)**(2*n))
            dft = cv2.dft(np.float32(arr), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            dft_shift[:,:,0] *= H
            dft_shift[:,:,1] *= H
            f_ishift = np.fft.ifftshift(dft_shift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            self.processed_image = Image.fromarray(img_back.astype(np.uint8)).convert("RGB")
            self.cv_processed = cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.display_images()

    def gaussian_freq_filter(self, mode):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            u = np.arange(rows)
            v = np.arange(cols)
            U, V = np.meshgrid(u, v, sparse=False, indexing='ij')
            D = np.sqrt((U-crow)**2 + (V-ccol)**2)
            D0 = 30
            if mode == "lpf":
                H = np.exp(-(D**2)/(2*(D0**2)))
            else:
                H = 1 - np.exp(-(D**2)/(2*(D0**2)))
            dft = cv2.dft(np.float32(arr), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            dft_shift[:,:,0] *= H
            dft_shift[:,:,1] *= H
            f_ishift = np.fft.ifftshift(dft_shift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            self.processed_image = Image.fromarray(img_back.astype(np.uint8)).convert("RGB")
            self.cv_processed = cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.display_images()

    def homomorphic_filter(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L")).astype(np.float32) + 1
            arr_log = np.log(arr)
            dft = cv2.dft(arr_log, flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            rows, cols = arr.shape
            crow, ccol = rows//2, cols//2
            H = np.ones((rows, cols, 2), np.float32)
            gammaL, gammaH, c, D0 = 0.5, 2.0, 1, 30
            for u in range(rows):
                for v in range(cols):
                    D = np.sqrt((u-crow)**2 + (v-ccol)**2)
                    H[u,v] = (gammaH - gammaL)*(1 - np.exp(-c*(D**2)/(D0**2))) + gammaL
            dft_shift *= H
            f_ishift = np.fft.ifftshift(dft_shift)
            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
            img_back = np.exp(img_back)
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            self.processed_image = Image.fromarray(img_back.astype(np.uint8)).convert("RGB")
            self.cv_processed = cv2.cvtColor(img_back.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.display_images()

    def sobel(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            sobelx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
            sobel = np.sqrt(sobelx**2 + sobely**2)
            sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
            self.processed_image = Image.fromarray(sobel.astype(np.uint8)).convert("RGB")
            self.cv_processed = cv2.cvtColor(sobel.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.display_images()

    def prewitt(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
            kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            prewittx = cv2.filter2D(arr, cv2.CV_64F, kernelx)
            prewitty = cv2.filter2D(arr, cv2.CV_64F, kernely)
            prewitt = np.sqrt(prewittx**2 + prewitty**2)
            prewitt = cv2.normalize(prewitt, None, 0, 255, cv2.NORM_MINMAX)
            self.processed_image = Image.fromarray(prewitt.astype(np.uint8)).convert("RGB")
            self.cv_processed = cv2.cvtColor(prewitt.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.display_images()

    def roberts_cross(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            kernelx = np.array([[1, 0], [0, -1]])
            kernely = np.array([[0, 1], [-1, 0]])
            robertsx = cv2.filter2D(arr, cv2.CV_64F, kernelx)
            robertsy = cv2.filter2D(arr, cv2.CV_64F, kernely)
            roberts = np.sqrt(robertsx**2 + robertsy**2)
            roberts = cv2.normalize(roberts, None, 0, 255, cv2.NORM_MINMAX)
            self.processed_image = Image.fromarray(roberts.astype(np.uint8)).convert("RGB")
            self.cv_processed = cv2.cvtColor(roberts.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.display_images()

    def compass(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            kernels = [
                np.array([[-1,-1,-1],[1,1,1],[-1,-1,-1]]),
                np.array([[-1,1,-1],[-1,1,-1],[-1,1,-1]]),
                np.array([[1,-1,-1],[1,1,-1],[1,1,1]]),
                np.array([[-1,-1,1],[-1,1,1],[1,1,1]])
            ]
            max_img = np.zeros_like(arr, dtype=np.float32)
            for k in kernels:
                filtered = cv2.filter2D(arr, -1, k)
                max_img = np.maximum(max_img, filtered)
            max_img = cv2.normalize(max_img, None, 0, 255, cv2.NORM_MINMAX)
            self.processed_image = Image.fromarray(max_img.astype(np.uint8)).convert("RGB")
            self.cv_processed = cv2.cvtColor(max_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.display_images()

    def canny(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            edges = cv2.Canny(arr, 100, 200)
            self.processed_image = Image.fromarray(edges).convert("RGB")
            self.cv_processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.display_images()

    def laplace(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            lap = cv2.Laplacian(arr, cv2.CV_64F)
            lap = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX)
            self.processed_image = Image.fromarray(lap.astype(np.uint8)).convert("RGB")
            self.cv_processed = cv2.cvtColor(lap.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            self.display_images()

    def gabor(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            kernel = cv2.getGaborKernel((21,21), 8.0, np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(arr, cv2.CV_8UC3, kernel)
            self.processed_image = Image.fromarray(filtered).convert("RGB")
            self.cv_processed = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
            self.display_images()

    def hough_transform(self):
        if self.image:
            arr = np.array(self.processed_image.convert("L"))
            edges = cv2.Canny(arr, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
            arr_color = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            if lines is not None:
                for rho,theta in lines[:,0]:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(arr_color,(x1,y1),(x2,y2),(0,0,255),2)
            self.processed_image = Image.fromarray(arr_color)
            self.cv_processed = arr_color
            self.display_images()

    def kmeans_segmentation(self):
        if self.image:
            arr = np.array(self.processed_image)
            Z = arr.reshape((-1,3))
            Z = np.float32(Z)
            K = simpledialog.askinteger("K-means", "Kaç küme?", minvalue=2, maxvalue=10)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            res2 = res.reshape((arr.shape))
            self.processed_image = Image.fromarray(res2)
            self.cv_processed = cv2.cvtColor(res2, cv2.COLOR_RGB2BGR)
            self.display_images()

    def erode(self):
        if self.image:
            ksize = simpledialog.askinteger("Erode", "Çekirdek boyutu (tek sayı):", minvalue=1, maxvalue=21)
            arr = np.array(self.processed_image)
            kernel = np.ones((ksize,ksize), np.uint8)
            eroded = cv2.erode(arr, kernel, iterations=1)
            self.processed_image = Image.fromarray(eroded)
            self.cv_processed = cv2.cvtColor(eroded, cv2.COLOR_RGB2BGR)
            self.display_images()

    def dilate(self):
        if self.image:
            ksize = simpledialog.askinteger("Dilate", "Çekirdek boyutu (tek sayı):", minvalue=1, maxvalue=21)
            arr = np.array(self.processed_image)
            kernel = np.ones((ksize,ksize), np.uint8)
            dilated = cv2.dilate(arr, kernel, iterations=1)
            self.processed_image = Image.fromarray(dilated)
            self.cv_processed = cv2.cvtColor(dilated, cv2.COLOR_RGB2BGR)
            self.display_images()

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()