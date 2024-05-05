# Final : ML-OPS | Diabetes Prediction Pipeline

### Name        :   Bilhaq Avi Dewantara
### Username    :   crossnexx
### Programme   :   ML-OPS with Cloudeka

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset) |
| Masalah | di Indonesia saat ini tidak sedikit orang-orang yang memiliki penyakit diabetes pada diri mereka, baik itu dikarenakan keturunan maupun dari pola kehidupannya sehari-hari. Untuk kasus tersebut tentunya perlu diwaspadai, apalagi orang Indonesia sangat suka sekali nyemil makanan (apalagi manis). Sehingga hal itu dapat menyebabkan kenaikan gula darah melebihi kebutuhan masing-masing individu per hari. Menurut Permenkes Nomor 30 Tahun 2013, anjuran konsumsi gula per orang per hari adalah 10% dari total energi (200 kkal) atau sekitar 4 gula sendok makan (50 gram) per harinya. Dengan adanya kenaikan gula darah yang berlebihan, hormon insulin tidak dapat membantu tubuh dalam mengontrol kadar gula darah dan menyebabkan diabetes. Salah satu penyakit kronis di Indonesia adalah <i>diabetes melitus</i> yang menyebabkan kematian tertinggi ke 3 di Indonesia tahun 2019 sebesar 57,42 kematian per 100.000 penduduk. Data dari <i>International Diabetes Federation</i> (IDF) menyatakan bahwa penderita diabetes 2021 di Indonesia meningkat pesat dalam 10 tahun terakhir dan diperkirakan 28,57 juta pada 2045 [[1]](https://ditpui.ugm.ac.id/diabetes-penyebab-kematian-tertinggi-di-indonesia-batasi-dengan-snack-sehat-rendah-gula/). Pada dataset yang digunakan berisi tentang kumpulan data-data medis pasien yang berstatus positif dan negatif, fitur-fitur yang ada seperti umur, jenis kelamin, BMI, hipertensi, penyakit jantung, riwayat rokok, kadar <i>HbA1c</i>, dan glukosa. Dengan adanya dataset ini bisa menjadi solusi dalam mengidentifikasi pasien yang berisiko terkena diabetes dan dapat dibuatkan rencana pengobatan khusus. |
| Solusi machine learning | Perlu adanya pendeteksi diabetes yang dapat diimplementasikan pada pasien dalam pengecekan kadar gula darah. Dengan begitu dapat mengetahui pola dari setiap individu yang memiliki pertanda penyakit diabetes. |
| Metode pengolahan | Metode pengolahan data yang dilakukan adalah sebanyak 9 kolom dengan 8 kolom fitur dan 1 kolom label pada dataset dilakukan <i>preprocessing</i> dengan transformasi untuk mengubah 2 fitur kategori (<i>string</i>) menjadi <i>one_hot</i> dan 6 fitur numerik menjadi <i>scaling</i> antara 0 - 1. Selanjutnya membagi dataset dengan 80% <i>data training</i>dan 20% <i>data evaluation</i>. |
| Arsitektur model | Arsitektur model yang digunakan sederhana dengan menggunakan <i>Dense Layer</i> dan <i>Dropout Layer</i> untuk <i>Hidden Layer</i> pada model <i>Neural Network</i>-nya, dan dilakukan juga <i>Learning Rate</i> yang diakhiri oleh <i>Output Layer</i>, semuanya dilakukan dengan mencari <i>best hyperparameters</i> yang dilakukan oleh tuner menggunakan algoritma <i>RandomSearch</i>. Model di <i>compile</i> dengan <i>Adam optimizers, loss</i> dengan <i>binary_crossentropy</i>, dan <i>metrics</i> dengan <i>BinaryAccuracy</i>. |
| Metrik evaluasi | Metric yang digunakan pada model yaitu <i>AUC, Precision, Recall, ExampleCount,</i> dan <i>BinaryAccuracy</i> untuk mengevaluasi performa model dalam menentukan prediksi nantinya |
| Performa model | Model yang telah dibuat ditinjau dari accuracy sebesar 0.96 pada proses training dan validation_accuracy sebesar 0.957. Kemudian berdasarkan <i>metrics evaluation</i> menghasilkan <i>Precision</i> dengan rerata 0.842, <i>Recall</i> sebesar 0.644, <i>AUC</i> sebesar 0.962, <i>ExampleCount</i> dengan nilai 19764, <i>BinaryAccuracy</i> sebesar 0.958, dan <i>Loss</i> sebesar 0.115. Sehingga dari nilai <i>metrics</i> tersebut, model ini dapat terbilang cukup baik dalam menjalankan fungsinya. |
| Opsi Deploy | Model yang telah di <i>training</i>, dilakukan <i>deployment</i> dengan menggunakan Cloudeka dengan layanan <i>Deka Flexi</i>. |
| <i>Web app</i> | <i>Deployment web</i> dapat dilihat pada link [Diabetes-Prediction](http://103.190.215.156:8501/v1/models/diabetes-prediction-model/metadata). |
| <i>Monitoring</i> | Model ini telah dilakukan proses <i>monitoring</i> yang menggunakan <i>Prometheus</i> sebagai dasarnya. Kemudian untuk menyederhanakan monitoring agar dapat dipahami oleh awam disini menggunkanan <i>Grafana</i>. Hal yang ditampilkan pada <i>dashboard Grafana</i> adalah <i>Request Count & Run Time</i>, yang mana pada saat melakukan <i>prediction request</i> oleh <i>user</i> dan berapa lama waktu model di jalankan. |