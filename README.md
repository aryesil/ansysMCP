# ansysMCP

**ANSYS 2025 R1 için gerçek çözücü süren bir MCP (Model Context Protocol) sunucusu.**
Claude Code, Claude Desktop, Cursor veya Google Antigravity gibi bir yapay zekâ istemcisi, doğal dille ANSYS Mechanical ve Fluent'i sürebilir: CAD dosyasını okur, analizi kurar, çözer, sonucu doğrular.

> **English summary.** ansysMCP is an MCP server that drives a *real* ANSYS 2025 R1 installation: Mechanical through PyMechanical, Fluent through PyFluent, and result files through PyDPF. It covers static, modal, buckling, harmonic, transient and random-vibration structural analysis, steady thermal analysis, aerodynamic CFD, layered composite shells, mesh-convergence studies and design-of-experiments planning. Nothing is simulated or made up: when the solver is missing or fails, the tool returns an error instead of a number. It needs Windows, a licensed ANSYS 2025 R1 and Python 3.10+. Tool descriptions and messages are written in Turkish.

---

## İçindekiler

1. [Ne yapar, ne yapmaz](#ne-yapar-ne-yapmaz)
2. [Gereksinimler](#gereksinimler)
3. [Kurulum](#kurulum)
4. [İstemciye bağlama](#i̇stemciye-bağlama)
5. [İlk analiz](#i̇lk-analiz)
6. [Yapılandırma](#yapılandırma)
7. [Araç kataloğu](#araç-kataloğu)
8. [Doğrulama durumu](#doğrulama-durumu)
9. [Tasarım ilkeleri](#tasarım-ilkeleri)
10. [Bilinen sınırlar](#bilinen-sınırlar)
11. [Sorun giderme](#sorun-giderme)
12. [Proje yapısı](#proje-yapısı)

---

## Ne yapar, ne yapmaz

**Yapar**

| Alan | Kapsam |
|------|--------|
| Yapısal (Mechanical) | Statik, modal, doğrusal burkulma, harmonik, geçici (transient), rastgele titreşim (PSD) |
| Yükler ve mesnetler | Kuvvet, basınç, moment, uzak kuvvet, yer çekimi, zorlanmış yer değiştirme, silindirik mesnet, yalnız-basınç mesnedi, ek mesnet ve yükler |
| Malzeme | Kütüphane kartları, izotrop ve ortotropik elastisite, termal özellikler |
| Kompozit | Yüzey (kabuk) gövdede katmanlı kesit (ply kalınlığı ve açısı), klasik laminat teorisi ile ön hesap |
| Isıl | Kararlı hal ısı iletimi, taşınım, ısıl gerilme değerlendirmesi |
| Aerodinamik (Fluent) | Sanal rüzgâr tüneli, prizma sınır tabakası (y+ hedefli), Cd / Cl / Cm, hücum açısı taraması, yan kayma, serbest akış yönü (STEP gövdeyi döndürerek) |
| Mesh | Üç kademeli mesh bağımsızlığı (GCI, ASME V&V 20), gerilme tekilliği teşhisi, mesh kalite kapısı |
| Montaj | Temas çiftleri (bonded, frictional, frictionless) |
| Sonuçlar | Kontur görselleri (PNG), düğüm bazında CSV/VTU dışa aktarma, Markdown/HTML rapor, el hesabı karşılaştırması |
| Tasarım keşfi | Latin Hypercube / full factorial deney tasarımı, Pareto cephesi (çözücü çağırmaz, dışarıdan gelen gerçek yanıtlarla çalışır) |
| Güvenlik | AST tabanlı üç seviyeli script sandbox'ı, SHA-256 provenance imzası ve tahrifat denetimi |

**Yapmaz**

* Çözücü yokken sahte sonuç üretmez; hata döner.
* ANSYS lisansı vermez. Kurulu ve lisanslı bir ANSYS gerekir.
* Katı CAD'den orta yüzey çıkarmaz; katmanlı kompozit için yüzey (kabuk) STEP gerekir.
* ACP (Composite PrepPost) iş akışı ve ply hasar ölçütleri yoktur.

---

## Gereksinimler

| Bileşen | Sürüm / not |
|---------|-------------|
| İşletim sistemi | Windows 10 / 11 (yollar, Mechanical ve Fluent süreçleri Windows için yazıldı) |
| ANSYS | **2025 R1 (v251)** önerilen sürümdür; 2025 R2, 2024 R2 ve 2024 R1 de aranır |
| ANSYS lisansı | Mechanical (yapısal / ısıl) ve/veya Fluent (CFD) çözümü için geçerli lisans |
| Python | 3.10 veya üstü (3.11 ile geliştirildi ve test edildi) |
| Disk | Kurulum ve iş çıktıları için birkaç GB boş alan (CFD ağı ve sonuç dosyaları büyüktür) |
| İsteğe bağlı | SolidWorks + `pywin32` (yalnızca `.sldprt` / `.sldasm` dosyalarını STEP'e çevirmek için) |

ANSYS kurulumu `AWP_ROOT251` ortam değişkeninden ya da `C:\Program Files\ANSYS Inc\v251` gibi standart klasörlerden bulunur.

---

## Kurulum

### 1. Depoyu indirin

```powershell
git clone https://github.com/aryesil/ansysMCP.git
cd ansysMCP
```

### 2. Sanal ortam oluşturun ve bağımlılıkları kurun

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` şunları kurar: `mcp`, `fastmcp`, `pydantic`, `ansys-mechanical-core` (PyMechanical), `ansys-fluent-core` (PyFluent), `ansys-dpf-core` (PyDPF), `ansys-geometry-core`, `pyvista`, `trimesh`, `cascadio` (STEP okuma), `h5py`, `networkx`, `scipy`, `matplotlib`, `numpy`.

SolidWorks dosyalarını dönüştürecekseniz ayrıca:

```powershell
pip install pywin32
```

### 3. Kurulumu doğrulayın

ANSYS'in bulunduğunu ve Python paketlerinin yüklendiğini kontrol edin:

```powershell
python -c "import sys; sys.path.insert(0, 'src'); from ansys_mcp.config import SystemConfig; print(SystemConfig.get_ansys_summary())"
```

Çıktıda `'detected': True` ve ANSYS sürümü görünmelidir. `False` ise [Sorun giderme](#sorun-giderme) bölümüne bakın.

Sunucunun kendisini elle başlatmak da mümkündür (stdio üzerinden konuşur, istemci olmadan sessiz bekler):

```powershell
python server.py
```

---

## İstemciye bağlama

Sunucunun giriş noktası depo kökündeki `server.py` dosyasıdır. İstemci bu dosyayı, sanal ortamdaki Python ile çalıştırır. Aşağıdaki örneklerde yolları kendi klasörünüze göre değiştirin.

> Bağlandıktan sonra `get_client_configurations` aracı, çalışan sunucunun kendi yollarıyla hazır yapılandırma bloklarını döndürür.

### Claude Code

```powershell
claude mcp add ansys -- C:\yol\ansysMCP\venv\Scripts\python.exe C:\yol\ansysMCP\server.py
```

Ya da proje `.mcp.json` dosyasına:

```json
{
  "mcpServers": {
    "ansys": {
      "type": "stdio",
      "command": "C:/yol/ansysMCP/venv/Scripts/python.exe",
      "args": ["C:/yol/ansysMCP/server.py"]
    }
  }
}
```

### Claude Desktop

`%APPDATA%\Claude\claude_desktop_config.json` dosyasına ekleyin:

```json
{
  "mcpServers": {
    "ansys-workbench": {
      "command": "C:/yol/ansysMCP/venv/Scripts/python.exe",
      "args": ["C:/yol/ansysMCP/server.py"],
      "cwd": "C:/yol/ansysMCP"
    }
  }
}
```

### Cursor

`~/.cursor/mcp.json` (ya da proje içi `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "ansys": {
      "command": "C:/yol/ansysMCP/venv/Scripts/python.exe",
      "args": ["C:/yol/ansysMCP/server.py"],
      "cwd": "C:/yol/ansysMCP"
    }
  }
}
```

### Google Antigravity

Antigravity'nin MCP ayarında aynı `command` / `args` çiftini kullanın.

Yapılandırmayı değiştirdikten sonra istemciyi yeniden başlatın; yeni araçlar ancak o zaman görünür.

---

## İlk analiz

İstemciye doğal dille anlatmanız yeterlidir. Örnek istek:

> "`C:\cad\boru.step` dosyasındaki karbon borunun bir ucu sabit, diğer ucuna 50 N yanal kuvvet uygulanıyor. Statik analiz yap ve uç sapmasını kapalı form çözümle karşılaştır."

Arka planda tipik araç sırası:

1. `get_ansys_environment_info`: ANSYS ve lisans durumu
2. `inspect_cad` / `inspect_geometry_details`: hacim, alan, sınır kutusu, yüz listesi
3. `create_analysis`: iş (job) kaydı oluşturur, `job_id` döner
4. `configure_static_analysis`: malzeme, mesh boyutu, mesnet ve yük (yüzeyler normal veya `min_x` / `max_x` gibi uç seçiciyle adlandırılır)
5. `validate_analysis`: girdi kapısı; eksik ya da fiziksel olmayan girdi burada reddedilir
6. `submit_analysis`: canlı Mechanical çözümü
7. `get_analysis_results`, `render_structural_contours`, `hand_calc_check`, `export_report`

Mechanical'ı ekranda izlemek için `show_gui=True` verin; pencere açık kalır ve her adım görülebilir.

Fluent CFD için akış: `configure_aerodynamics_cfd` ardından `solve_cfd_simulation` (tek nokta) veya `solve_cfd_polar` (hücum açısı taraması).

---

## Yapılandırma

| Ayar | Anlamı | Varsayılan |
|------|--------|------------|
| `AMPC_RUNS_DIR` | İş klasörlerinin ve iş veritabanının (`orchestrator.db`) yeri | `<depo>/runs` |
| `AWP_ROOT251` (veya 252 / 242 / 241) | ANSYS kurulum klasörü | Otomatik aranır (`C:\Program Files\ANSYS Inc\v251` ...) |

* Mechanical, yerel gRPC portu **10000** üzerinden sürülür. Sunucu bu portta dinleyen bir Mechanical'a bağlanır; yoksa yenisini başlatır.
* Tek paylaşılan Mechanical oturumu vardır; uzun çözümler birbirini beklemez, biri koşarken ikincisi "meşgul" hatası alır. Aynı makinede elle açılmış bir Workbench oturumu aynı oturumu etkileyebilir.
* `runs/` klasörü (iş çıktıları, `.rst`, `.cas.h5`, görseller, CAD kopyaları) yerel veridir ve depoya girmez.

Ortam değişkenini istemci yapılandırmasında verebilirsiniz:

```json
"env": { "AMPC_RUNS_DIR": "E:/ansys_runs" }
```

---

## Araç kataloğu

Sunucu 56 araç ve 9 kaynak (resource) sunar.

**Oturum ve ortam**
`get_ansys_environment_info`, `open_ansys_session`, `close_ansys_session`, `get_live_session_steps`, `get_client_configurations`

**İş yönetimi**
`create_analysis`, `validate_analysis`, `list_jobs`, `get_job_status`, `clone_job`, `delete_job`, `cancel_job`

**Geometri ve CAD**
`inspect_cad`, `inspect_geometry_details`, `inspect_live_geometry_faces`, `find_semantic_faces`, `render_geometry_preview`, `convert_cad_to_step` (SolidWorks gerekir)

**Yapısal analiz**
`configure_static_analysis` + `submit_analysis`,
`configure_modal_analysis` + `solve_modal_analysis`,
`configure_buckling_analysis` + `solve_buckling_analysis`,
`configure_harmonic_analysis` + `solve_harmonic_analysis`,
`configure_transient_analysis` + `solve_transient_analysis`,
`configure_random_vibration_analysis` + `solve_random_vibration_analysis`,
`configure_assembly_contacts`

**Isıl**
`configure_thermal_analysis` + `solve_thermal_analysis`, `evaluate_thermal_stresses`, `render_thermal_contours`

**Aerodinamik / CFD**
`configure_aerodynamics_cfd`, `solve_cfd_simulation`, `solve_cfd_polar`, `get_cfd_aerodynamic_coefficients`

**Mesh ve doğruluk**
`run_mesh_convergence_study`, `detect_stress_singularities`, `get_static_equilibrium_status`, `hand_calc_check`

**Sonuç ve rapor**
`get_analysis_results`, `render_structural_contours`, `render_modal_mode_shapes`, `export_result_field`, `export_report`, `export_mechanical_script`, `evaluate_resonance_risk`

**Malzeme ve birim**
`lookup_material`, `convert_units`

**Optimizasyon (DoE)**
`generate_optimization_design_points`, `submit_design_point_results`, `evaluate_optimization_study`, `get_pareto_optimal_designs`

**Güvenlik**
`execute_safe_python_script`, `verify_analysis_provenance`

**Kaynaklar (resources)**
`ansys://job/{job_id}/manifest`, `.../logs`, `.../validation`, `.../mesh-convergence`, `.../cfd-monitors`, `.../provenance`, `.../modal-frequencies`, `.../thermal-results`, `ansys://optimization/{study_id}/pareto-front`

Her aracın parametreleri ve ayrıntıları, istemcinin araç listesinde açıklama olarak görünür.

---

## Doğrulama durumu

Çözüm yolları gerçek ANSYS 2025 R1'e karşı, kapalı form çözümlerle karşılaştırılarak denenmiştir. Örnek gövde: 16/14 mm x 585 mm ankastre karbon boru (E = 135 GPa).

| Analiz | Sonuç | Referans |
|--------|-------|----------|
| Statik, uçtan 50 N | uç sapması 18.574 mm | 18.57 mm (`F L³ / 3EI`) |
| Statik reaksiyon | 50.000 N | 50 N |
| Modal | 1. eğilme frekansı 79.79 Hz | 79.8 Hz |
| Geçici (basamak yük) | tepe 0.7101 mm | 0.7198 mm (sönümlü tek mod) |
| Rastgele titreşim | düz PSD yanıtı | Miles formülü, %6 içinde |
| Katmanlı kabuk (tek yönlü) | uç sapması 15.54 mm | 15.36 mm (laminat teorisi) |
| Katmanlı kabuk (çapraz 0/90/90/0) | 28.52 mm | 28.47 mm |
| Katmanlı kabukta modal, burkulma, harmonik, geçici, PSD | kapalı formlarla uyumlu | ince cidarlı konsol |
| CFD, ters ve çapraz akış | Cd, referans aralığında | silindir kesiti, Re ~ 3e4 |

Bu sayılar geliştirme makinesindeki ölçümlerdir; sizin lisansınız, mesh boyutunuz ve geometrinizle farklı çıkabilir. Doğrulanmamış yollar [Bilinen sınırlar](#bilinen-sınırlar) bölümünde açıkça yazılıdır.

---

## Tasarım ilkeleri

* **Uydurma yok.** Çözücü çalışmıyorsa, lisans yoksa ya da çözüm başarısızsa araç hata döndürür. Sayı üretmek için varsayılan değer kullanılmaz.
* **Açık girdi.** Malzeme, mesh boyutu, mesnet yüzeyi, yük, sönüm, PSD gibi fiziksel girdilerin hepsi kullanıcıdan alınır. Eksik veya fiziksel olmayan girdi `validate_analysis` aşamasında reddedilir.
* **Doğrulanabilirlik.** Her çözüm denge kontrolü (uygulanan yük + reaksiyon = 0) ve isteğe bağlı el hesabı karşılaştırması ile birlikte raporlanır. Çözülmüş sonuçlar `.rst` dosyasından okunur.
* **İzlenebilirlik.** Her iş, girdi ve sonuçların SHA-256 imzasını içeren bir provenance kaydı taşır; `verify_analysis_provenance` ile tahrifat denetlenir.
* **Güvenli script çalıştırma.** `execute_safe_python_script`, AST politikasından geçen kodu ayrı, yalıtılmış bir yorumlayıcıda zaman sınırıyla çalıştırır. Yüksek seviye için açık onay belirteci ister.

---

## Bilinen sınırlar

* **Yalnızca Windows** hedeflenmiştir.
* Katmanlı kompozit **yüzey (kabuk) gövde** ister. Katı CAD'den orta yüzey çıkarma yoktur; katı gövdeyle çağrılırsa hata döner.
* **Ply hasar ölçütleri ve ACP iş akışı yoktur.** Katmanlı kesitte gerilme aracı ve emniyet katsayısı geçersiz olduğundan eklenmez; toplam deplasman, von Mises ve reaksiyon raporlanır.
* Serbest akış yönü STEP gövde için canlı doğrulanmıştır. STL/OBJ/PLY ve montaj STEP yolları kodlanmıştır ama Fluent ile canlı denenmemiştir.
* Parasolid / SAT dosyalarıyla CFD denenmemiştir.
* Mechanical tek paylaşılan oturumla çalışır; paralel çözüm desteklenmez.
* Araç açıklamaları ve hata mesajları Türkçedir.

---

## Sorun giderme

| Belirti | Olası neden / çözüm |
|---------|---------------------|
| `get_ansys_summary()` `detected: False` | `AWP_ROOT251` ortam değişkenini kurulum klasörüne (`C:\Program Files\ANSYS Inc\v251`) ayarlayın |
| "Canlı Mechanical oturumu yok" | `open_ansys_session` çağırın veya ANSYS lisans sunucusunun açık olduğunu kontrol edin |
| Çözüm sırasında bağlantı koptu | Aynı anda başka bir Mechanical / Workbench oturumu port 10000'i kullanıyor olabilir; kapatıp yeniden deneyin. Sunucu bir sonraki çağrıda yeni oturum açar |
| Araç "meşgul" diyor | Başka bir çözüm koşuyor; bitmesini bekleyin |
| İstemcide araçlar görünmüyor | `server.py` ve Python yolunu kontrol edin; istemciyi tamamen yeniden başlatın |
| İlk geometri çağrısı çok yavaş | Ağır kütüphaneler (VTK, scipy, OpenCASCADE) başlangıçta yüklenir; sunucu açıldıktan sonra tekrar deneyin |
| CFD çözümü uzun sürüyor | Ağ boyutunu büyütün (`mesh_size_mm`) ya da iterasyon sayısını düşürün; gerçek Fluent çözümleri dakikalar sürer |
| Disk doluyor | `AMPC_RUNS_DIR` ile iş klasörünü başka bir sürücüye taşıyın, eski işleri `delete_job` ile silin |

---

## Proje yapısı

```text
ansysMCP/
├── server.py                  # Giriş noktası (ağır kütüphaneleri ana iş parçacığında yükler)
├── pyproject.toml
├── requirements.txt
└── src/ansys_mcp/
    ├── server.py              # MCP araçları ve kaynakları
    ├── config.py              # ANSYS kurulum tespiti
    ├── preload.py             # Yerel kütüphanelerin başlangıçta yüklenmesi
    ├── adapters/              # Mechanical (canlı), Fluent, DPF, sonuç dışa aktarma, görselleştirme
    ├── analysis/              # GCI, mesh çalışması, gerilme tekilliği
    ├── cfd/                   # Rüzgâr tüneli, prizma tabakası, hacim ağı
    ├── composites/            # Klasik laminat teorisi
    ├── domain/                # Veri modelleri, birimler, malzeme kütüphanesi, el hesapları
    ├── geometry/              # CAD ayrıştırma, semantik yüz eşleme, STEP döndürme, temaslar
    ├── modal/                 # Modal motor, rezonans değerlendirmesi, mod şekilleri
    ├── optimization/          # DoE ve Pareto
    ├── orchestrator/          # İş durum makinesi, SQLite depolama, süreç yönetimi
    ├── security/              # Sandbox, provenance, istemci yapılandırma üretici
    ├── thermal/               # Isıl gerilme ve görselleştirme
    └── validation/            # Girdi kapıları, yük ve mesnet şemaları, denge denetimi
```

---

## Lisans

Bu depoda henüz bir lisans dosyası yoktur. Kodu kullanmadan, değiştirmeden veya dağıtmadan önce depo sahibine danışın. ANSYS, Mechanical, Fluent ve ilgili adlar ANSYS, Inc.'in tescilli markalarıdır; bu proje ANSYS ile bağlantılı değildir ve ANSYS tarafından onaylanmamıştır.
