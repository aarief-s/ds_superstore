
# Sales Revenue Prediction: Mendorong Profitabilitas Ritel dengan Data Science

## Ikhtisar Proyek

Proyek ini bertujuan untuk mengatasi tantangan umum dalam sektor ritel di mana tingginya volume penjualan tidak selalu berbanding lurus dengan profitabilitas yang sehat. Hal ini seringkali disebabkan oleh strategi diskon yang tidak efektif dan adanya produk yang merugi. Dengan memanfaatkan metodologi CRISP-DM (Cross-Industry Standard Process for Data Mining) dan teknik Machine Learning, proyek ini bertujuan untuk memberikan *insight* prediktif dan rekomendasi strategis untuk meningkatkan margin profit.

## Masalah Bisnis

Identifikasi masalah bisnis spesifik meliputi:
* Profit margin rendah atau bahkan negatif pada kategori produk tertentu
* Variasi performa penjualan dan profitabilitas antar wilayah (region)
* Perbedaan profit antar segmen pelanggan
* Strategi diskon yang tidak efektif
* Keberadaan produk *slow-moving* dan *high-demand*

## Tujuan Proyek

* **Mengidentifikasi Faktor Pendorong Profit:** Menemukan variabel-variabel kunci yang berkorelasi kuat dengan profitabilitas
* **Mengurangi Transaksi Rugi:** Meminimalkan transaksi yang menghasilkan kerugian, dengan target di bawah 5% dari total transaksi
* **Meningkatkan Profit Margin Keseluruhan:** Meningkatkan *overall profit margin* minimal 15%
* **Memberikan *Insight* Real-Time:** Menyediakan pemahaman yang dapat ditindaklanjuti secara cepat
* **Membangun Model Prediktif:** Mengembangkan model Machine Learning untuk memprediksi penjualan di masa depan
* **Mengidentifikasi Tren dan Pola:** Mendeteksi tren dan pola penjualan serta profitabilitas yang signifikan

## Pertanyaan Bisnis

* Bagaimana memastikan profitabilitas tetap sehat meskipun penjualan tinggi, di tengah tantangan diskon tidak efektif, produk merugi, serta variasi profit antar kategori, wilayah, dan segmen pelanggan?
* Apakah prediksi menggunakan model Machine Learning dapat memberikan *insight* dan solusi bisnis yang tepat untuk perusahaan?
  
## Kumpulan Data

Dataset yang digunakan bersumber dari **Kaggle Superstore Dataset**, berisi 9.994 baris (transaksi) dan 21 kolom (fitur). Fitur-fitur ini dikelompokkan menjadi:
* **Identitas Unik:** Order ID, Customer ID, Product ID
* **Dimensi Waktu:** Order Date, Ship Date
* **Dimensi Lokasi:** Country, State, City, Region
* **Dimensi Produk:** Category, Sub-Category, Product Name
* **Dimensi Pelanggan:** Customer Name, Segment
* **Metrik Transaksi (Numerik):** Sales, Quantity, Discount, Profit

### Feature Engineering

Beberapa fitur baru direkayasa untuk meningkatkan kekuatan prediktif model, termasuk:
* Fitur Komponen Waktu
* Fitur Efisiensi Pengiriman
* Fitur RFM (Recency, Frequency, Monetary)
* Fitur Metrik Keuangan
* Fitur Tingkat Diskon
* Fitur Kinerja Konstektual

## Persiapan Data

* **Data Cleaning:** Penanganan *missing value* dan *duplicated data*, serta penghapusan kolom yang tidak relevan
* **Ukuran Data Final:** Setelah persiapan, dataset memiliki 8731 baris dan 57 kolom

## Eksplorasi Data Analisis (EDA) & *Key Findings*

EDA mengungkapkan beberapa *insight* penting:

* **Profit Margin Distribusi:** Terdapat dua puncak utama pada distribusi profit margin: sekitar 5% dan 50%. Mean profit margin sekitar 30%, menunjukkan adanya "penarik" ke bawah dari transaksi berprofit rendah
* **Efisiensi Pengiriman:**
    * Mode "Same Day" adalah yang tercepat (rata-rata kurang dari 1 hari)
    * "First Class" adalah tercepat kedua (rata-rata 2 hari)
    * "Standard Class" adalah yang terlambat (rata-rata 5 hari)
    * **Rekomendasi:** Promosikan "Same Day" dan "First Class" sebagai opsi premium
* **Segmen Pelanggan:** Segmen "Champions" memiliki jumlah transaksi tertinggi, mengindikasikan mereka adalah pelanggan paling aktif dan loyal
    * **Rekomendasi:** Berikan perlakuan istimewa, penawaran eksklusif, dan program loyalitas untuk segmen "Champions"
* **Kategori Kecepatan Pengiriman:** Mayoritas pengiriman (52.4%) berada dalam kategori "Medium", dengan 38.4% di "Fast"
    * **Rekomendasi:** Ada potensi besar untuk mengoptimalkan rute agar sebagian besar pengiriman "Medium" beralih ke "Fast"
* **Tren Penjualan Mingguan:** Peningkatan penjualan yang konsisten dari Jumat hingga mencapai puncaknya pada hari Minggu
    * **Rekomendasi:** Intensifkan kampanye pemasaran atau promosi khusus pada hari Sabtu dan Minggu untuk memaksimalkan pendapatan
* **Profitabilitas Berdasarkan Region:** Region Central mencatat profit tertinggi secara signifikan, diikuti oleh Region South
    * **Rekomendasi:** Selidiki strategi dan karakteristik pasar di Region Central dan South
* **Korelasi Sales dan Profit:** Secara umum, terdapat korelasi positif antara Sales dan Profit; semakin tinggi penjualan, cenderung semakin tinggi pula keuntungannya
    * **Rekomendasi:** Identifikasi karakteristik transaksi dengan penjualan dan profit tinggi untuk replikasi faktor keberhasilan

## Pemodelan Machine Learning

Beberapa model regresi dievaluasi untuk memprediksi penjualan:
* Linear Regression 
* Ridge Regression 
* Lasso Regression 
* XGBoost
* Random Forest 

### Evaluasi Model

| Model             | Train R2 | Test R2 | Train RMSE | Test RMSE | Train MAE | Test MAE | Cv_rmse_Mean | Cv_rmse_std | Overfitting_R2 | Overfitting_RMSE |
| :---------------- | :------- | :------ | :--------- | :-------- | :-------- | :-------- | :----------- | :---------- | :------------- | :--------------- |
| Linear Regression | 0.7593   | 0.7722  | 234.7606   | 249.2844  | 123.2867  | 123.3084  | 235.0429     | 52.6183     | -0.0130        | 14.5238          |
| Ridge Regression  | 0.7593   | 0.7722  | 234.7606   | 249.2965  | 123.2625  | 123.2823  | 235.0427     | 52.6248     | -0.0129        | 14.5359          |
| Lasso Regression  | 0.7593   | 0.7722  | 234.7617   | 249.2732  | 123.1848  | 123.1992  | 235.0349     | 52.6392     | -0.0130        | 14.5114          |
| Random Forest     | 0.9982   | 0.9504  | 20.0768    | 116.3352  | 0.8319    | 3.3626    | 54.4865      | 49.2657     | 0.0478         | 96.2583          |
| XGBoost           | 0.9996   | 0.9349  | 9.1343     | 133.2238  | 4.7783    | 10.3797   | 67.3256      | 31.3137     | 0.0647         | 124.0894         |

Grafik perbandingan metrik menunjukkan:
* **R² Score Comparison:** Random Forest dan XGBoost menunjukkan performa terbaik dengan R² score mendekati sempurna (~1.0), sedangkan model regresi linear memiliki performa sedang (~0.75-0.8)
* **RMSE Comparison:** Random Forest dan XGBoost menunjukkan performa terbaik dengan RMSE sangat rendah (Train ~20, Test ~115-130), sementara model regresi linear memiliki error yang jauh lebih tinggi (~230-250)
* **Cross-Validation RMSE:** Random Forest dan XGBoost menunjukkan performa terbaik dan konsisten dengan CV RMSE rendah (~55-65) dan variabilitas kecil
* **Overfitting Analysis (R²):** Random Forest dan XGBoost menunjukkan sedikit *overfitting* yang masih dapat diterima, sedangkan model regresi linear menunjukkan *underfitting*

## Kesimpulan Model

* **Model Terbaik:** Model **Random Forest** teridentifikasi sebagai model terbaik untuk memprediksi penjualan
* **Akurasi Tinggi:** Model ini mencapai akurasi prediksi yang sangat tinggi dengan **Test R² sebesar 0.9504**
* **Tingkat Kesalahan:** Tingkat kesalahan prediksi rata-rata (RMSE) model adalah **$116.34**, menunjukkan keandalan yang baik
* **Generalisasi Baik:** Model menunjukkan tingkat *overfitting* yang rendah, yang berarti mampu generalisasi dengan baik pada data baru

## Interpretasi Model (Fitur Penting)

Model Random Forest mengidentifikasi faktor-faktor paling dominan yang memengaruhi total penjualan:
1.  **Revenue per Quantity (58.67%)**: Pendapatan per unit produk adalah faktor paling dominan yang mempengaruhi total penjualan
2.  **Quantity (24.62%)**: Jumlah produk per transaksi menjadi faktor kedua terpenting
3.  **High Value Transaction (12.91%)**: Status transaksi bernilai tinggi memberikan kontribusi tambahan dalam prediksi
   
## Rekomendasi Bisnis

Berdasarkan analisis dan hasil model, rekomendasi strategis meliputi:
1.  **Prioritaskan Kategori & Wilayah Unggulan:** Alokasikan sumber daya untuk "Office Supplies" terutama di wilayah "Central" untuk memaksimalkan penjualan dan keuntungan
2.  **Promosikan Opsi Pengiriman Cepat:** Tingkatkan promosi opsi pengiriman "Same Day" dan "First Class" untuk kepuasan dan loyalitas pelanggan
3.  **Program Loyalitas untuk "Champions":** Berikan program loyalitas eksklusif untuk mempertahankan segmen pelanggan "Champions" yang merupakan aset terbesar perusahaan
4.  **Dorong "Loyal Customers" & "Potential Loyalists":** Berikan penawaran dan rekomendasi yang dipersonalisasi untuk mendorong pembelian berulang
5.  **Insentif untuk "New Customers":** Berikan insentif khusus kepada "New Customers" untuk mendorong pembelian berulang sejak awal
6.  **Manfaatkan Model Prediktif:** Gunakan model Random Forest untuk perencanaan inventaris, alokasi sumber daya, dan penetapan target penjualan di masa depan
7.  **Diskon Strategis:** Terapkan diskon secara strategis hanya pada produk atau momen yang terbukti meningkatkan profitabilitas secara keseluruhan
8.  **Tingkatkan *Revenue per Unit*:** Fokus pada peningkatan rata-rata pendapatan per unit produk melalui strategi harga atau inisiatif *cross-selling/up-selling*




