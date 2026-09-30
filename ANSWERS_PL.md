# Pytania rekrutacyjne z Machine Learning – wyczerpujące odpowiedzi (PL)

> Odpowiedzi na wszystkie pytania z `machine-learning-interview-questions/README.md`, w tej samej kolejności i pod tymi samymi kategoriami. Każda odpowiedź kończy się listą źródeł.

## Spis treści


### [Inżynieria AI](#inżynieria-ai)


### [Podstawy uczenia maszynowego](#podstawy-uczenia-maszynowego)

- [1. Wyjaśnij pojęcia epoch, batch, batch size i iteration.](#q1)
- [2. Czym są embeddingi (embeddings) w uczeniu maszynowym?](#q2)
- [3. Czym jest funkcja aktywacji softmax?](#q3)
- [4. Czym jest uczenie maszynowe (Machine Learning)?](#q4)
- [5. Czym różni się uczenie nadzorowane od nienadzorowanego?](#q5)
- [6. Czym jest uczenie ze wzmocnieniem (Reinforcement Learning)?](#q6)
- [7. Czym jest bias?](#q7)
- [8. Jaka jest różnica między klasyfikacją a regresją?](#q8)
- [9. Wyjaśnij overfitting i underfitting. Jak im zapobiegać?](#q9)
- [10. Czym są funkcje straty L1 i L2?](#q10)
- [11. Czym jest regularyzacja? Wyjaśnij regularyzację L1 (Lasso) i L2 (Ridge).](#q11)
- [12. Czym są funkcje straty (loss function) i funkcje kosztu (cost function)? Jaka jest kluczowa różnica?](#q12)
- [13. Czym jest dropout?](#q13)
- [14. Czym jest perceptron?](#q14)
- [15. Wyjaśnij wielowarstwowy perceptron (MLP – Multilayer Perceptron).](#q15)
- [16. Czym jest cross-entropy?](#q16)
- [17. Czym są logity (logits)?](#q17)
- [18. Wyjaśnij cross-validation. Dlaczego się ją stosuje?](#q18)
- [19. Czym są precision, recall i F1-score?](#q19)
- [20. Czym jest wykrywanie anomalii (anomaly detection)?](#q20)
- [21. Jaka jest różnica między metodami policy-based i value-based?](#q21)
- [22. Czym jest Q-Learning?](#q22)
- [23. Wyjaśnij koncepcję exploration vs exploitation.](#q23)
- [24. Wyjaśnij przekleństwo wymiarowości (curse of dimensionality) i jak sobie z nim radzić.](#q24)
- [25. Wyjaśnij Local Loss, Focal Loss i Gradient Blending w kontekście Multi-Task Learning.](#q25)
- [26. Wyjaśnij Contrastive Learning.](#q26)
- [27. Czym jest Generative AI?](#q27)

### [Algorytmy](#algorytmy)

- [28. Jak działa algorytm drzewa decyzyjnego (Decision Tree)?](#q28)
- [29. Wyjaśnij, jak drzewa decyzyjne dokonują podziałów i obsługują cechy kategoryczne.](#q29)
- [30. Jak działa Random Forest? Jak ulepsza drzewa decyzyjne i jak redukuje wariancję?](#q30)
- [31. Wyjaśnij metody zespołowe (Ensemble Methods). Dlaczego są tak skuteczne?](#q31)
- [32. Jaka jest różnica między baggingiem a boostingiem?](#q32)
- [33. Czym jest Gradient Boosting? Jak działa XGBoost?](#q33)
- [34. Jakie są kluczowe hiperparametry XGBoost?](#q34)
- [35. Wyjaśnij Gradient Boosting i jego zalety w porównaniu z Random Forests.](#q35)
- [36. Wyjaśnij, czym regresja logistyczna różni się od regresji liniowej.](#q36)
- [37. Jak działa regresja logistyczna?](#q37)
- [38. Wyjaśnij R-kwadrat i skorygowany R-kwadrat.](#q38)
- [39. Jak sprawdzić współliniowość (multicollinearity) w modelach regresji?](#q39)
- [40. Jak działa algorytm K-najbliższych sąsiadów (KNN)?](#q40)
- [41. Wyjaśnij K-Means Clustering. Jak działa? Jakie ma ograniczenia?](#q41)
- [42. Wyjaśnij Support Vector Machines (SVM). Czym jest kernel trick?](#q42)
- [43. Czym jest granica decyzyjna (decision boundary) w klasyfikatorach?](#q43)
- [44. Wyjaśnij Naive Bayes.](#q44)
- [45. Czym jest redukcja wymiarowości (Dimensionality Reduction)?](#q45)
- [46. Wyjaśnij PCA (Principal Component Analysis). Jak działa? Kiedy go użyć?](#q46)
- [47. Wyjaśnij Gradient Descent i jego warianty.](#q47)
- [48. Czym jest krzywa ROC-AUC i jak ją interpretować?](#q48)

### [Przygotowanie danych i inżynieria cech](#przygotowanie-danych-i-inżynieria-cech)

- [49. Czym jest Feature Engineering?](#q49)
- [50. Czym jest kodowanie one-hot (one-hot encoding)? Kiedy go stosować?](#q50)
- [51. Jak radzić sobie z brakującymi danymi?](#q51)
- [52. Jak radzić sobie z wartościami odstającymi (outliers)?](#q52)
- [53. Wyjaśnij skalowanie cech (Feature Scaling). Dlaczego jest potrzebne?](#q53)
- [54. One-Hot, Label, Target i K-Fold Target Encoding - na czym polegają i czym się różnią?](#q54)
- [55. Jak radzić sobie z cechami kategorycznymi?](#q55)
- [56. Selekcja cech (feature selection) a ekstrakcja cech (feature extraction) - jaka jest różnica?](#q56)
- [57. Jak tworzyć nowe cechy z istniejących?](#q57)
- [58. Jak podejść do zbioru danych z silnie niezbalansowanymi klasami?](#q58)
- [59. Jak wybierasz cechy dla modelu?](#q59)
- [60. Dlaczego i jak dzielimy dane na zbiór treningowy, testowy i walidacyjny?](#q60)

### [Optymalizacja](#optymalizacja)

- [61. Czym jest gradient descent? Jak działa?](#q61)
- [62. Czym jest stochastic gradient descent (SGD)?](#q62)
- [63. Czym są zanikające gradienty (vanishing gradients)?](#q63)
- [64. Czym jest learning rate? Jak dobrać dobry?](#q64)
- [65. Jak learning rate wpływa na trening modelu?](#q65)
- [66. Jak podejść do strojenia hiperparametrów (hyperparameter tuning)?](#q66)
- [67. Czym jest kwantyzacja modelu (model quantization) i kiedy ją stosować?](#q67)
- [68. Jak zapewnić sprawiedliwość (fairness) i ograniczyć bias w modelach ML?](#q68)
- [69. Czym różnią się Grid Search, Random Search i optymalizacja bayesowska (Bayesian Optimization)?](#q69)
- [70. Wyjaśnij optymalizację hiperparametrów metodą TPE (Tree-structured Parzen Estimator).](#q70)
- [71. Wyjaśnij optymalizację bayesowską (Bayesian Optimization).](#q71)
- [72. Wyjaśnij optymalizator Adam.](#q72)
- [73. Wyjaśnij optymalizator RMSprop.](#q73)
- [74. Czym jest optymalizator Adagrad?](#q74)

### [Deep Learning](#deep-learning)

- [75. Czym są sieci neuronowe (neural networks)?](#q75)
- [76. Wyjaśnij sieć neuronową typu feedforward (Feedforward Neural Network).](#q76)
- [77. Czym są propagacja w przód (forward propagation) i propagacja wsteczna (backward propagation)?](#q77)
- [78. Czym jest propagacja wsteczna (backpropagation)?](#q78)
- [79. Wymień i wyjaśnij kilka hiperparametrów używanych do trenowania sieci neuronowej.](#q79)
- [80. Jaka jest przewaga głębokiego uczenia nad tradycyjnym uczeniem maszynowym?](#q80)
- [81. Czym są funkcje aktywacji i dlaczego się je stosuje?](#q81)
- [82. Wyjaśnij funkcje aktywacji Sigmoid, Tanh, ReLU, LeakyReLU i Softmax wraz z ich zaletami i wadami.](#q82)
- [83. Dlaczego Sigmoid i Tanh nie są preferowane w warstwach ukrytych sieci neuronowej?](#q83)
- [84. Czym jest dropout i dlaczego jest skuteczny?](#q84)
- [85. Jaki jest wpływ dropoutu na szybkość treningu i inferencji?](#q85)
- [86. Czym jest regularyzacja L1/L2 i jak wpływa na sieć neuronową?](#q86)
- [87. Czym jest normalizacja wsadowa (batch normalization) i dlaczego się ją stosuje?](#q87)
- [88. Jakie hiperparametry normalizacji wsadowej (batch normalization) można optymalizować?](#q88)
- [89. Czym jest współdzielenie parametrów (parameter sharing) w deep learningu?](#q89)
- [90. Czym jest uczenie reprezentacji (representation learning) i dlaczego jest użyteczne?](#q90)
- [91. Czym jest model generatywny i czym różni się od modelu dyskryminatywnego?](#q91)
- [92. Wyjaśnij, jak działa model generatywny.](#q92)
- [93. Wyjaśnij architekturę koder-dekoder (Encoder-Decoder).](#q93)
- [94. Jaka jest różnica między architekturami Transformer typu encoder-only, decoder-only i encoder-decoder?](#q94)
- [95. Czym jest przestrzeń ukryta (latent space)?](#q95)
- [96. Czym są autoenkodery (autoencoders)? Wyjaśnij ich warstwy i praktyczne zastosowania.](#q96)
- [97. Czym jest autoenkoder wariacyjny (VAE) i czym różni się od tradycyjnego autoenkodera?](#q97)
- [98. W jaki sposób VAE narzuca probabilistyczną strukturę przestrzeni latentnej i dlaczego jest to ważne?](#q98)
- [99. Jaka jest architektura sieci GAN (Generative Adversarial Network)?](#q99)
- [100. Jakie role pełnią generator i dyskryminator w sieci GAN?](#q100)
- [101. Czym jest zapadanie trybów (mode collapse) w GAN i jak można je łagodzić?](#q101)
- [102. W jaki sposób GAN-y są używane w syntezie obrazów lub zadaniach tłumaczenia obraz-do-obrazu (image-to-image translation)?](#q102)
- [103. Czym są konwolucyjne sieci neuronowe (Convolutional Neural Networks, CNN)?](#q103)
- [104. Czym są filtry (kernels) w CNN?](#q104)
- [105. Czym jest stride w CNN?](#q105)
- [106. Czym jest padding w CNN?](#q106)
- [107. Czym jest pooling w CNN?](#q107)
- [108. Czym są warstwy w pełni połączone (fully connected layers) w CNN?](#q108)
- [109. Czym jest rekurencyjna sieć neuronowa (Recurrent Neural Network, RNN)?](#q109)
- [110. Jakie są ograniczenia RNN i jak się je rozwiązuje?](#q110)
- [111. Czym są LSTM i GRU? Jak rozwiązują problem zależności długoterminowych?](#q111)
- [112. Jakie są główne bramki w LSTM i jakie pełnią role?](#q112)
- [113. Jak zidentyfikować problem eksplodujących gradientów w modelu?](#q113)
- [114. Czym jest architektura Transformer i co odróżnia ją od CNN i RNN?](#q114)
- [115. Czym jest mechanizm uwagi (Attention) w deep learningu i dlaczego jest istotny?](#q115)
- [116. Jaka jest podstawowa różnica między LSTM a Transformerami?](#q116)
- [117. Czym są modele dyfuzyjne (Diffusion Models)?](#q117)
- [118. Dlaczego dyfuzja działa lepiej niż autoregresja?](#q118)
- [119. Wyjaśnij transfer learning i kiedy go stosować.](#q119)
- [120. Czym są modele multimodalne (Multimodal AI) i jak przetwarzają różne typy danych?](#q120)
- [121. Jak działają modele świata (World Models)?](#q121)
- [122. Jak działają dyfuzyjne modele językowe (Diffusion Language Models, DLM)?](#q122)
- [123. Wyjaśnij Deep RL from Human Preferences (uczenie ze wzmocnieniem na podstawie ludzkich preferencji).](#q123)

### [NLP](#nlp)

- [124. Jakie są zalety Transformerów nad tradycyjnymi modelami sequence-to-sequence?](#q124)
- [125. Jakie są ograniczenia Transformerów i jak można je adresować?](#q125)
- [126. Czym jest BERT i jak poprawia rozumienie języka?](#q126)
- [127. Jak trenuje się Transformery (pretraining i fine-tuning)?](#q127)
- [128. Wyjaśnij transfer learning w kontekście Transformerów.](#q128)
- [129. Opisz proces generowania tekstu w modelach językowych opartych na Transformerach.](#q129)
- [130. Czym są modele Seq2Seq?](#q130)
- [131. Porównaj modele N-gram i modele deep learningowe (kompromisy).](#q131)
- [132. Czym jest model n-gram?](#q132)
- [133. Czym jest TF-IDF i czym różni się od word embeddings?](#q133)
- [134. Czym jest Bag-of-Words?](#q134)
- [135. Do czego służy perplexity w NLP?](#q135)
- [136. Czym różni się stemming od lematyzacji?](#q136)
- [137. Czym jest Latent Semantic Indexing (LSI)?](#q137)
- [138. Czym jest dependency parsing (analiza zależnościowa)?](#q138)
- [139. Jakie są podejścia do streszczania tekstu (text summarization)?](#q139)
- [140. Czym są word embeddings (zanurzenia słów)?](#q140)
- [141. Czym jest Word2Vec?](#q141)
- [142. Czym jest t-SNE i jak jest używane w NLP?](#q142)
- [143. Wyjaśnij ColBERT](#q143)

### [Computer Vision (widzenie komputerowe)](#computer-vision-widzenie-komputerowe)

- [144. Czym jest computer vision i dlaczego jest ważne?](#q144)
- [145. Czym jest segmentacja obrazu i jakie są jej zastosowania?](#q145)
- [146. Czym jest detekcja obiektów i czym różni się od klasyfikacji obrazów?](#q146)
- [147. Jakie są kroki budowy systemu rozpoznawania obrazów?](#q147)
- [148. Jakie są wyzwania w śledzeniu obiektów w czasie rzeczywistym (real-time object tracking)?](#q148)
- [149. Czym jest ekstrakcja cech (feature extraction) w computer vision?](#q149)
- [150. Czym jest OCR i jakie są jego główne zastosowania?](#q150)
- [151. Czym CNN różni się od tradycyjnych sieci neuronowych w computer vision?](#q151)
- [152. Czym jest augmentacja danych (data augmentation) i jakie techniki są powszechnie stosowane?](#q152)
- [153. Jakie są popularne frameworki deep learning dla computer vision?](#q153)
- [154. Jak Transformery mogą być używane w zadaniach computer vision?](#q154)

### [Duże modele językowe (LLM)](#duże-modele-językowe-llm)

- [155. Czym jest duży model językowy (LLM) i jak działa?](#q155)
- [156. Czym jest architektura Transformer i jak działa?](#q156)
- [157. Jakie są kluczowe komponenty architektury Transformer?](#q157)
- [158. Czego uczy się każdy blok Transformera (What Each Transformer Block Learns)?](#q158)
- [159. Dlaczego skalujemy iloczyn skalarny w attention przez √dₖ?](#q159)
- [160. Czym jest KV Cache w LLM?](#q160)
- [161. Czym jest Paged Attention w LLM?](#q161)
- [162. Czym jest Flash Attention?](#q162)
- [163. Czym jest speculative decoding i jak przyspiesza inferencję?](#q163)
- [164. Jak continuous batching poprawia przepustowość inferencji LLM?](#q164)
- [165. Czym różni się pre-training od fine-tuningu w LLM?](#q165)
- [166. Jakie są wyzwania w trenowaniu LLM?](#q166)
- [167. Czym jest zero-shot learning w kontekście LLM?](#q167)
- [168. Jak radzić sobie z uprzedzeniami (bias) i sprawiedliwością (fairness) w LLM?](#q168)
- [169. Jakie są rzeczywiste zastosowania LLM w biznesie i technologii?](#q169)
- [170. W jaki sposób architektura Transformer poprawia wydajność LLM względem RNN?](#q170)
- [171. Wyjaśnij Query (Q), Key (K) i Value (V) w mechanizmie attention.](#q171)
- [172. Czym jest self-attention i jak działa w Transformerach?](#q172)
- [173. Czym jest Cross Attention w Transformerach?](#q173)
- [174. Czym są mechanizmy multi-head attention? Dlaczego używa się wielu głowic uwagi?](#q174)
- [175. W jaki sposób attention pomaga uchwycić zależności dalekiego zasięgu (long-range dependencies)?](#q175)
- [176. Czym jest Grouped-Query Attention (GQA) i czym różni się od Multi-Head Attention (MHA)?](#q176)
- [177. Czym są sieci Feed-Forward (FFN) w LLM?](#q177)
- [178. Tokenizacja w dużych modelach językowych (LLM).](#q178)
- [179. Czym jest tokenizacja podsłowna (subword tokenization)?](#q179)
- [180. Czym jest BPE (Byte Pair Encoding) w LLM?](#q180)
- [181. Czym jest positional embedding w LLM?](#q181)
- [182. Czym jest temperatura (temperature) w kontekście LLM?](#q182)
- [183. Czym jest maskowanie przyczynowe (causal masking)?](#q183)
- [184. Czym są skip connections (połączenia rezydualne)?](#q184)
- [185. Jak działa Rotary Position Embedding (RoPE) i dlaczego jest preferowane nad uczonymi positional embeddings?](#q185)
- [186. Czym jest normalizacja (normalization)? Wyjaśnij RMSNorm (Root Mean Square Layer Normalization).](#q186)
- [187. Czym jest dropout i jak jest stosowany w LLM?](#q187)
- [188. Dlaczego Attention używa Softmax?](#q188)
- [189. Co przechowuje baza wektorowa (Vector DB) w zastosowaniach LLM?](#q189)
- [190. Jak poprawić szybkość inferencji w produkcyjnych wdrożeniach LLM?](#q190)
- [191. Czym jest okno kontekstowe (context window) w LLM?](#q191)
- [192. Dlaczego okno kontekstowe jest ograniczone w LLM?](#q192)
- [193. Wyjaśnij Prompting, Retrieval-Augmented Generation (RAG) i Fine-Tuning.](#q193)
- [194. Czym jest Mixture of Experts (MoE) i jak działa w modelach takich jak Mixtral?](#q194)
- [195. Jaka jest różnica między modelami gęstymi (dense) a rzadkimi (sparse)?](#q195)
- [196. Transformery pracują na tekście – czy potrafią też rozumieć obrazy?](#q196)
- [197. Małe modele językowe (Small Language Models, SLM).](#q197)
- [198. Duże modele rozumujące (Large Reasoning Models, LRM).](#q198)
- [199. Uczenie ze wzmocnieniem z ludzkiej informacji zwrotnej (RLHF).](#q199)
- [200. Proximal Policy Optimization (PPO).](#q200)
- [201. Direct Preference Optimization (DPO).](#q201)
- [202. Group Relative Policy Optimization (GRPO).](#q202)
- [203. Rekursywne modele językowe (Recursive Language Models, RLM).](#q203)
- [204. Uczenie ciągłe (Continual Learning) w LLM.](#q204)
- [205. Jak działa destylacja wiedzy (Knowledge Distillation)?](#q205)
- [206. Czym jest instruction tuning i dlaczego jest ważny dla modeli czatowych?](#q206)
- [207. Prefill vs Decode: czym się różnią fazy inferencji LLM?](#q207)
- [208. Jak działa Sliding Window Attention?](#q208)
- [209. Jak działają Attention Sinks?](#q209)

### [Ewaluacja modeli](#ewaluacja-modeli)

- [210. Czym są precision, recall, F1 score i accuracy?](#q210)
- [211. Czym jest macierz pomyłek (confusion matrix) i jak ją interpretować?](#q211)
- [212. Jakie są typowe metryki ewaluacji w klasyfikacji?](#q212)
- [213. Kiedy używać accuracy, a kiedy innych metryk?](#q213)
- [214. Kiedy używać log loss zamiast accuracy?](#q214)
- [215. Jakich metryk użyjesz w klasyfikacji wieloklasowej?](#q215)
- [216. Jak radzić sobie z niezbalansowaniem klas w metrykach klasyfikacji?](#q216)
- [217. Czym jest krzywa ROC? Czym jest AUC?](#q217)
- [218. Jak radzić sobie z niezbalansowanymi zbiorami danych?](#q218)
- [219. Jakie są typowe metryki ewaluacji w regresji?](#q219)
- [220. Jaka jest różnica między MAE, MSE i RMSE?](#q220)
- [221. Jak wybrać właściwą metrykę ewaluacji dla danego problemu?](#q221)
- [222. Jak porównać wydajność różnych modeli?](#q222)
- [223. Wyjaśnij walidację krzyżową (cross-validation) i jej znaczenie.](#q223)
- [224. Czym jest strojenie hiperparametrów (Hyperparameter Tuning)?](#q224)
- [225. Jak ocenia się modele uczenia nienadzorowanego?](#q225)
- [226. Jak ocenić algorytm klasteryzacji?](#q226)
- [227. Jakich metryk użyjesz dla systemu rekomendacyjnego?](#q227)
- [228. Czym jest A/B testing w kontekście ML?](#q228)
- [229. Czym jest LLM as a Judge?](#q229)

### [Projektowanie systemów i MLOps](#projektowanie-systemów-i-mlops)

- [230. Zaprojektuj agenta głosowego AI działającego w czasie rzeczywistym (Real-Time Voice AI Agent)](#q230)
- [231. Zaprojektuj ChatGPT: od treningu do serwowania (end to end)](#q231)
- [232. Zaprojektuj system RAG (rozmowa z własnymi dokumentami)](#q232)
- [233. Zaprojektuj pamięć dla osobistego asystenta AI](#q233)
- [234. Zaprojektuj agenta Deep Research](#q234)
- [235. Zaprojektuj wieloagentowy system obsługi klienta (Multi-Agent Customer Support)](#q235)
- [236. Zaprojektuj asystenta AI działającego na urządzeniu (On-Device AI Assistant)](#q236)
- [237. Zaprojektuj multimodalny system wyszukiwania (tekst, obraz, wideo)](#q237)
- [238. Zaprojektuj platformę inferencji LLM (vLLM-as-a-Service)](#q238)
- [239. Zaprojektuj platformę do ewaluacji LLM (LLM Evaluation Platform)](#q239)
- [240. Zaprojektuj usługę generowania obrazów z tekstu (text-to-image, w stylu Midjourney)](#q240)
- [241. Zaprojektuj usługę generowania muzyki (w stylu Suno)](#q241)
- [242. Zaprojektuj usługę generowania wideo (w stylu Sora)](#q242)
- [243. Zaprojektuj agenta programistycznego AI (AI Coding Agent)](#q243)
- [244. Zaprojektuj system ML do rekomendacji filmów na YouTube](#q244)
- [245. Zaprojektuj system ML do wyszukiwania filmów na YouTube](#q245)
- [246. Zaprojektuj system ML do spersonalizowanego feedu treści](#q246)
- [247. Zaprojektuj system ML do wykrywania szkodliwych treści (harmful content detection)](#q247)
- [248. Zaprojektuj system ML do rekomendacji podobnych ofert (Similar Listings) na Airbnb](#q248)
- [249. Zaprojektuj system ML do rekomendacji produktów zastępczych (Replacement Product Recommendation)](#q249)
- [250. Zaprojektuj system ML do rekomendacji wydarzeń (Event Recommendation)](#q250)
- [251. Zaprojektuj system ML do wyszukiwania multimodalnego (Multimodal Search)](#q251)
- [252. Zaprojektuj system ML do przewidywania kliknięć w reklamy (Ad Click Prediction)](#q252)
- [253. Zaprojektuj system ML do szacowania czasu dostawy (Estimate Delivery Time)](#q253)
- [254. Zaprojektuj system ML do wyszukiwania obrazów (Image Search)](#q254)
- [255. Zaprojektuj system ML do rekomendacji znajomych (Friends Recommendation)](#q255)
- [256. Zaprojektuj system rekomendacji produktów dla platformy e-commerce](#q256)
- [257. Jak zbudowałbyś system wykrywania nadużyć finansowych (fraud detection)?](#q257)
- [258. Techniki fuzji multimodalnej w uczeniu maszynowym: early fusion vs late fusion](#q258)
- [259. Jak podejść do problemu prognozowania szeregów czasowych (time series forecasting)?](#q259)
- [260. Jak zbudowałbyś system wykrywania spamu?](#q260)
- [261. Opisz, jak zaimplementowałbyś system klasyfikacji obrazów](#q261)
- [262. Jakie podejście zastosowałbyś w zadaniu analizy sentymentu?](#q262)
- [263. Jak zaprojektowałbyś model predykcji odejścia klientów (customer churn)?](#q263)
- [264. Jak podszedłbyś do rankingu wyników wyszukiwania?](#q264)
- [265. Jak zbudowałbyś system wykrywania anomalii w ruchu sieciowym?](#q265)
- [266. Jak wybrać właściwy algorytm uczenia maszynowego?](#q266)
- [267. Czym jest dryf modelu (model drift) i jak sobie z nim radzić?](#q267)
- [268. Jak poradzisz sobie z trenowaniem na danych wielkoskalowych (large-scale data)?](#q268)
- [269. Jak radzisz sobie z zaszumionymi danymi (noisy data) w modelach ML?](#q269)
- [270. Jakie strategie zastosujesz, by skrócić czas trenowania modelu deep learningowego?](#q270)
- [271. Jak wdrożyć model ML na produkcję?](#q271)
- [272. Jak monitorować wydajność modelu na produkcji?](#q272)
- [273. Jak wdrożyć model o rygorystycznych wymaganiach dotyczących opóźnień (low-latency)?](#q273)
- [274. Jakie są typowe wyzwania przy wdrażaniu modeli ML?](#q274)
- [275. Omów wymagania dotyczące skalowalności i opóźnień w systemach ML.](#q275)
- [276. Jak zapewnić, że model jest skalowalny i dobrze działa na dużych zbiorach danych?](#q276)
- [277. Czym jest explainability modelu (wyjaśnialność)? Dlaczego jest ważna?](#q277)
- [278. Jakich technik użyłbyś, aby model był bardziej interpretowalny?](#q278)
- [279. Opisz swoje podejście do debugowania niedziałającego (słabo działającego) modelu ML.](#q279)
- [280. Jak zapewnić sprawiedliwość (fairness) i ograniczyć bias w modelach ML?](#q280)
- [281. Wyjaśnij MLOps i jego kluczowe komponenty.](#q281)
- [282. Czym jest feature store i dlaczego jest ważny?](#q282)
- [283. Wdrożenie modelu w chmurze vs on-device.](#q283)
- [284. Omów techniki kompresji modeli (Model Compression).](#q284)

### [Prawdopodobieństwo i statystyka](#prawdopodobieństwo-i-statystyka)

- [285. Wyjaśnij kompromis między obciążeniem a wariancją (Bias-Variance Tradeoff).](#q285)
- [286. Wyjaśnij różne rozkłady prawdopodobieństwa (normalny, dwumianowy, Poissona, jednostajny).](#q286)
- [287. Czym jest rozkład normalny i jego funkcje?](#q287)
- [288. Czym jest rozkład wykładniczy?](#q288)
- [289. Czym jest rozkład dwumianowy (binomial)?](#q289)
- [290. Czym jest rozkład Bernoulliego?](#q290)
- [291. Czym jest rozkład wielomianowy (multinomial)?](#q291)
- [292. Czym jest rozkład logarytmiczno-normalny (lognormal)?](#q292)
- [293. Czym jest rozkład logistyczny?](#q293)
- [294. Czym jest rozkład gamma i jego funkcje?](#q294)
- [295. Rozkład Poissona i jego funkcja.](#q295)
- [296. Kiedy użyć rozkładu Poissona zamiast dwumianowego?](#q296)
- [297. Czym jest wariancja?](#q297)
- [298. Czym jest odchylenie standardowe (stddev)?](#q298)
- [299. Wyjaśnij różnicę między średnią, medianą i modą.](#q299)
- [300. Jaka jest różnica między korelacją a kowariancją?](#q300)
- [301. Co oznacza współczynnik korelacji +1, 0 i -1?](#q301)
- [302. Wyjaśnij korelację a przyczynowość (Correlation vs Causation).](#q302)
- [303. Czym są błędy typu I i typu II?](#q303)
- [304. Czym jest p-value? Co oznacza istotność statystyczna?](#q304)
- [305. Wyjaśnij p-value i jego ograniczenia.](#q305)
- [306. Czym jest testowanie hipotez i kiedy jest używane w ML?](#q306)
- [307. Jakich testów statystycznych użyjesz do porównania dwóch modeli?](#q307)
- [308. Jak ocenić, czy cecha (feature) jest istotna statystycznie?](#q308)
- [309. Czym jest przedział ufności (confidence interval) i jak się go stosuje?](#q309)
- [310. Czym są z-score i t-score? Kiedy stosuje się każdy z nich?](#q310)
- [311. Wyjaśnij twierdzenie Bayesa. Jak odnosi się do Naive Bayes i metod bayesowskich?](#q311)
- [312. Jaka jest różnica między estymacją MLE a MAP?](#q312)
- [313. Czym jest estymacja największej wiarygodności (Maximum Likelihood Estimation, MLE)?](#q313)
- [314. Wyjaśnij podejście bayesowskie a częstościowe (Bayesian vs Frequentist) w statystyce.](#q314)
- [315. Czym jest centralne twierdzenie graniczne (Central Limit Theorem, CLT) i dlaczego jest ważne?](#q315)
- [316. Czym są techniki próbkowania (sampling techniques)?](#q316)
- [317. Czym jest metoda bootstrap i jak się ją stosuje?](#q317)

### [Kodowanie](#kodowanie)

- [318. Napisz funkcję w Pythonie obliczającą błąd średniokwadratowy (MSE).](#q318)
- [319. Napisz funkcję w Pythonie obliczającą średni błąd bezwzględny (MAE).](#q319)
- [320. Zaimplementuj prosty model regresji liniowej od zera.](#q320)
- [321. Zaimplementuj prosty model regresji logistycznej od zera.](#q321)
- [322. Zaimplementuj algorytm K najbliższych sąsiadów (K-Nearest Neighbors, KNN).](#q322)
- [323. Zaimplementuj funkcje aktywacji Sigmoid, Tanh, ReLU, LeakyReLU i Softmax.](#q323)
- [324. Jak zaimplementowałbyś klasteryzację k-means?](#q324)
- [325. Napisz kod wykonujący k-krotną walidację krzyżową (k-fold cross-validation).](#q325)
- [326. Jak użyjesz Pandas do wczytania i oczyszczenia danych?](#q326)
- [327. Zaimplementuj k-najbliższych sąsiadów (KNN) od zera.](#q327)
- [328. Napisz kod obliczający precision i recall.](#q328)

### [Matematyka](#matematyka)

- [329. Wartości własne i wektory własne (Eigenvalues and Eigenvectors)](#q329)

### [Pytania behawioralne i scenariuszowe](#pytania-behawioralne-i-scenariuszowe)

- [330. Opisz sytuację, w której poprawiłeś/aś wydajność modelu.](#q330)
- [331. Jak podszedłbyś do projektu z ograniczoną ilością danych z etykietami?](#q331)
- [332. Co zrobisz, jeśli model działa dobrze w testach, ale słabo na produkcji?](#q332)
- [333. Jak jesteś na bieżąco z postępami w ML?](#q333)
- [334. Opowiedz o wymagającym projekcie ML, nad którym pracowałeś/aś. Jaki był cel? Jaka była Twoja rola? Z jakimi wyzwaniami się mierzyłeś/aś? Jak je pokonałeś/aś? Jaki był wynik? Czego się nauczyłeś/aś?](#q334)
- [335. Dokąd zmierzają ML/AI w ciągu najbliższych 5 lat?](#q335)
- [336. Dlaczego interesuje Cię ta rola/firma?](#q336)
- [337. Opisz sytuację, w której Twój model zawiódł lub nie osiągnął oczekiwanych wyników. Co zrobiłeś/aś?](#q337)
- [338. Jak postąpisz w przypadku nieporozumień z kolegami dotyczących wyboru modeli lub podejść?](#q338)

---

## Inżynieria AI

**Przegląd:**

- **LLM (Large Language Model)** – duży model językowy oparty zwykle na architekturze Transformer, trenowany na ogromnych korpusach tekstu do przewidywania kolejnego tokenu (next-token prediction). Po pre-trainingu jest dostrajany (instruction tuning, RLHF/DPO), aby wykonywać polecenia. Ograniczenia: halucynacje, skończone okno kontekstu, wiedza „zamrożona” w momencie treningu.
- **RAG (Retrieval-Augmented Generation)** – wzorzec, w którym przed generacją odpowiedzi system wyszukuje w zewnętrznej bazie wiedzy (zwykle wektorowej, na podstawie embeddingów) fragmenty dokumentów i dołącza je do promptu. Zmniejsza halucynacje, pozwala korzystać z aktualnych i prywatnych danych bez ponownego treningu oraz cytować źródła. Kluczowe elementy: chunking, embeddingi, retriever, reranking, ewaluacja (faithfulness, recall).
- **MCP (Model Context Protocol)** – otwarty protokół standaryzujący sposób, w jaki aplikacje AI łączą się z narzędziami i źródłami danych (serwery MCP udostępniają tools, resources i prompts, a klient/host – np. aplikacja z LLM – z nich korzysta). Działa jak „uniwersalne złącze”, eliminując pisanie osobnych integracji dla każdej pary model–narzędzie.
- **Agent** – system, w którym LLM w pętli planuje, wywołuje narzędzia (tool calling), obserwuje wyniki i decyduje o kolejnych krokach aż do osiągnięcia celu (np. wzorzec ReAct). Dodatkowe elementy: pamięć, planowanie, guardrails, ewaluacja trajektorii. Wyzwania: niezawodność, koszt, opóźnienia, bezpieczeństwo (prompt injection).
- **Fine-tuning** – dalsze trenowanie wstępnie wytrenowanego modelu na węższym zbiorze danych, aby dostosować styl, format lub wiedzę domenową. Wariant pełny aktualizuje wszystkie wagi; metody PEFT (np. LoRA, QLoRA) uczą tylko małe adaptery, co mocno obniża koszt pamięci. Zwykle najpierw próbuje się prompt engineeringu i RAG, a fine-tuning stosuje, gdy trzeba zmienić zachowanie modelu.
- **Quantization** – redukcja precyzji liczb w wagach (a czasem aktywacjach) modelu, np. z FP16/BF16 do INT8 lub 4-bit. Zmniejsza zużycie pamięci i często przyspiesza inferencję kosztem niewielkiej, zwykle akceptowalnej utraty jakości. Popularne podejścia: post-training quantization (GPTQ, AWQ), quantization-aware training, QLoRA (fine-tuning na modelu 4-bit).

**Źródła:**
- [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization (wideo Outcome School)](https://www.youtube.com/watch?v=lnfWvX66FUk)
- [AI Engineering Interview Questions and Answers (repozytorium Amita Shekhara)](https://github.com/amitshekhariitbhu/ai-engineering-interview-questions)
- [Lewis i in., Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Model Context Protocol – oficjalna strona i dokumentacja](https://modelcontextprotocol.io)
- [Hu i in., LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Dettmers i in., QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)

---

## Podstawy uczenia maszynowego

<a id="q1"></a>
### 1. Wyjaśnij pojęcia epoch, batch, batch size i iteration.

**Odpowiedź:**

Podczas trenowania sieci neuronowej (i ogólnie modeli uczonych gradientowo) nie przetwarzamy całego zbioru naraz, tylko dzielimy go na porcje. Stąd cztery pojęcia:

- **Epoch (epoka)** – jedno pełne przejście algorytmu uczącego przez cały zbiór treningowy. Każda próbka została użyta do aktualizacji wag (średnio) raz.
- **Batch** – podzbiór próbek zbioru treningowego przetwarzany jednocześnie w jednym kroku (forward + backward).
- **Batch size** – liczba próbek w batchu (np. 32, 64, 256). To hiperparametr.
- **Iteration (iteracja, krok)** – jedna aktualizacja wag na podstawie jednego batcha.

Zależność: 

$$\text{iteracje na epokę} = \left\lceil \frac{N}{\text{batch size}} \right\rceil$$

**Przykład:** zbiór 10 000 próbek, batch size 100 → 100 iteracji na epokę; trening przez 20 epok to 2000 aktualizacji wag.

#### Warianty gradient descent
- **Batch GD** – batch size = N (cały zbiór): stabilny, dokładny gradient, ale wolny i pamięciożerny.
- **Stochastic GD** – batch size = 1: szumny gradient, częste aktualizacje.
- **Mini-batch GD** – kompromis, standard w praktyce.

#### Wpływ batch size
- Mały batch: więcej szumu w gradiencie (działa jak regularyzacja, często lepsza generalizacja), mniejsze zużycie pamięci GPU, wolniejsze wykorzystanie sprzętu.
- Duży batch: lepsza przepustowość na GPU, stabilniejszy gradient, ale ryzyko gorszej generalizacji i potrzeba skalowania learning rate (np. reguła liniowego skalowania z warm-upem).
- Gdy batch nie mieści się w pamięci, stosuje się **gradient accumulation** (sumowanie gradientów z kilku mini-batchy przed aktualizacją).

#### Ile epok?
Zbyt mało → underfitting, zbyt dużo → overfitting. W praktyce używa się **early stopping** na zbiorze walidacyjnym oraz harmonogramów learning rate.

```python
for epoch in range(num_epochs):
    for xb, yb in train_loader:          # każda iteracja = jeden batch
        loss = criterion(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Źródła:**
- [Epoch, Batch, Batch Size, Iteration (wideo Outcome School)](https://www.youtube.com/watch?v=NFLlXE-6vno)
- [Goodfellow, Bengio, Courville – Deep Learning, rozdz. 8: Optimization for Training Deep Models](https://www.deeplearningbook.org/contents/optimization.html)
- [PyTorch – Training a classifier (tutorial)](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

---

<a id="q2"></a>
### 2. Czym są embeddingi (embeddings) w uczeniu maszynowym?

**Odpowiedź:**

**Embedding** to gęsta, niskowymiarowa reprezentacja wektorowa obiektu (słowa, zdania, obrazu, użytkownika, produktu, węzła grafu), w której geometria przestrzeni odzwierciedla podobieństwo semantyczne: obiekty podobne leżą blisko siebie, a różnice między nimi mogą odpowiadać kierunkom w przestrzeni.

#### Dlaczego nie one-hot?
One-hot encoding dla słownika 50 000 słów daje wektory rzadkie o wymiarze 50 000, w których wszystkie słowa są jednakowo odległe – brak informacji o podobieństwie. Embedding (np. 256–1024 wymiarów) jest zwarty i uczy się znaczenia.

#### Jak powstają
- **Warstwa Embedding** w sieci: macierz $E \in \mathbb{R}^{V \times d}$, wiersz to wektor tokenu; uczona razem z resztą sieci przez backpropagation.
- **Word2Vec / GloVe** – uczone na współwystępowaniu słów (skip-gram, CBOW).
- **Modele kontekstowe** (BERT, sentence-transformers, modele embeddingowe LLM) – wektor zależy od kontekstu.
- **Modele multimodalne** (CLIP) – wspólna przestrzeń dla tekstu i obrazów.
- **Contrastive learning** – zbliża pary podobne, oddala niepodobne.

#### Miary podobieństwa
Najczęściej cosine similarity:

$$\cos(u,v)=\frac{u\cdot v}{\|u\|\,\|v\|}$$

lub iloczyn skalarny/odległość euklidesowa.

#### Zastosowania
Wyszukiwanie semantyczne i RAG (bazy wektorowe, ANN – np. HNSW), systemy rekomendacyjne, klasteryzacja, klasyfikacja z użyciem transfer learningu, wizualizacja (t-SNE, UMAP), wykrywanie duplikatów.

#### Pułapki
- Embeddingi dziedziczą uprzedzenia z danych.
- Nie są porównywalne między różnymi modelami – zmiana modelu wymaga ponownego indeksowania.
- Wymiar kompromisem między jakością a kosztem pamięci/wyszukiwania.

```python
import torch.nn as nn
emb = nn.Embedding(num_embeddings=50_000, embedding_dim=256)
vecs = emb(token_ids)   # (batch, seq_len, 256)
```

**Źródła:**
- [Embeddings in Machine Learning (wideo Outcome School)](https://www.youtube.com/watch?v=LedXW6xl21s)
- [Mikolov i in., Efficient Estimation of Word Representations in Vector Space (word2vec)](https://arxiv.org/abs/1301.3781)
- [Radford i in., Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Google ML Crash Course – Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings)

---

<a id="q3"></a>
### 3. Czym jest funkcja aktywacji softmax?

**Odpowiedź:**

**Softmax** zamienia wektor dowolnych liczb rzeczywistych (logitów) $z=(z_1,\dots,z_K)$ na rozkład prawdopodobieństwa nad $K$ klasami:

$$\text{softmax}(z)_i=\frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Własności: każde wyjście jest w $(0,1)$, suma wynosi 1, a funkcja jest monotoniczna (największy logit daje największe prawdopodobieństwo). Jest gładkim, różniczkowalnym przybliżeniem argmax.

#### Przykład
$z=(2.0,\,1.0,\,0.1)$ → $e^z\approx(7.39,\,2.72,\,1.11)$, suma $\approx 11.21$ → prawdopodobieństwa $\approx(0.66,\,0.24,\,0.10)$.

#### Stabilność numeryczna
Bezpośrednie liczenie $e^{z_i}$ może przepełnić zakres. Softmax jest niezmienniczy na przesunięcie o stałą, więc odejmuje się $\max(z)$:

$$\text{softmax}(z)_i=\frac{e^{z_i-\max(z)}}{\sum_j e^{z_j-\max(z)}}$$

W praktyce łączy się softmax z log i loss (log-softmax / `CrossEntropyLoss` w PyTorch przyjmuje surowe logity).

#### Temperatura
$\text{softmax}(z/T)$: $T>1$ spłaszcza rozkład (większa losowość, np. sampling w LLM), $T<1$ zaostrza.

#### Gradient
Dla cross-entropy z one-hot etykietą $y$: $\partial L/\partial z_i = p_i - y_i$ – bardzo prosty i stabilny.

#### Kiedy stosować
- Klasyfikacja wieloklasowa (klasy wzajemnie wykluczające się) – warstwa wyjściowa.
- Mechanizm uwagi (attention) – wagi uwagi.
- Dla klasyfikacji wieloetykietowej używa się **sigmoid** dla każdej klasy niezależnie.

#### Ograniczenia
Wyjścia bywają nadmiernie pewne (miscalibration); softmax zawsze „musi” wybrać klasę, nawet dla danych spoza rozkładu (out-of-distribution). Dla bardzo dużych słowników bywa kosztowny (stąd hierarchical/sampled softmax).

```python
import torch
p = torch.softmax(torch.tensor([2.0, 1.0, 0.1]), dim=0)
```

**Źródła:**
- [Softmax Activation Function in Machine Learning (wideo Outcome School)](https://www.youtube.com/watch?v=2Zx6x01WwWM)
- [Deep Learning book, rozdz. 6.2.2.3: Softmax Units](https://www.deeplearningbook.org/contents/mlp.html)
- [Wikipedia – Softmax function](https://en.wikipedia.org/wiki/Softmax_function)
- [PyTorch – torch.nn.Softmax](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html)

---

<a id="q4"></a>
### 4. Czym jest uczenie maszynowe (Machine Learning)?

**Odpowiedź:**

**Uczenie maszynowe** to dziedzina, w której algorytmy uczą się zależności z danych, zamiast być jawnie zaprogramowane regułami. Klasyczna definicja Toma Mitchella: program uczy się z doświadczenia $E$ względem zadania $T$ i miary jakości $P$, jeśli jego wyniki w $T$ mierzone $P$ poprawiają się wraz z $E$.

#### Idea
Zamiast pisać reguły „jeśli–to”, dostarczamy przykłady, a algorytm dobiera parametry modelu $f_\theta$ tak, by minimalizować funkcję straty na danych:

$$\theta^*=\arg\min_\theta \frac{1}{N}\sum_{i=1}^N \ell\big(f_\theta(x_i),y_i\big)$$

Celem nie jest zapamiętanie danych treningowych, lecz **generalizacja** – dobre działanie na nowych, niewidzianych danych.

#### Główne paradygmaty
- **Uczenie nadzorowane** (supervised) – dane z etykietami: klasyfikacja, regresja.
- **Uczenie nienadzorowane** (unsupervised) – brak etykiet: klasteryzacja, redukcja wymiaru, wykrywanie anomalii.
- **Uczenie częściowo nadzorowane i samonadzorowane** (semi-/self-supervised) – np. pre-training LLM.
- **Uczenie ze wzmocnieniem** (reinforcement learning) – agent uczy się z nagród.

#### Typowy pipeline
1. Sformułowanie problemu i metryk.
2. Zbieranie i czyszczenie danych, podział train/validation/test.
3. Inżynieria cech.
4. Wybór i trening modelu, strojenie hiperparametrów.
5. Ewaluacja na zbiorze testowym.
6. Wdrożenie i monitorowanie (drift danych).

#### Kluczowe trudności
Overfitting/underfitting (kompromis bias–variance), jakość i reprezentatywność danych, wyciek danych (data leakage), interpretowalność, uprzedzenia (bias) i koszty obliczeniowe.

#### Kiedy ML ma sens
Gdy reguły są zbyt złożone lub nieznane, dane są dostępne, a środowisko się zmienia. Gdy problem da się rozwiązać prostą, deterministyczną regułą, ML jest zbędne.

**Źródła:**
- [What is Machine Learning? (Outcome School)](https://outcomeschool.com/blog/machine-learning)
- [Wikipedia – Machine learning](https://en.wikipedia.org/wiki/Machine_learning)
- [Stanford CS229 – Machine Learning (materiały)](https://cs229.stanford.edu/)
- [scikit-learn – User Guide](https://scikit-learn.org/stable/user_guide.html)

---

<a id="q5"></a>
### 5. Czym różni się uczenie nadzorowane od nienadzorowanego?

**Odpowiedź:**

| Cecha | Nadzorowane (supervised) | Nienadzorowane (unsupervised) |
|---|---|---|
| Dane | pary $(x, y)$ z etykietami | tylko $x$, bez etykiet |
| Cel | nauczyć się mapowania $x \to y$ | odkryć strukturę w danych |
| Zadania | klasyfikacja, regresja | klasteryzacja, redukcja wymiaru, estymacja gęstości, wykrywanie anomalii |
| Ewaluacja | metryki względem prawdziwych etykiet (accuracy, RMSE, F1) | trudniejsza: silhouette, inertia, ocena ekspercka, zadania pochodne |
| Przykłady algorytmów | regresja liniowa/logistyczna, drzewa, SVM, sieci neuronowe | k-means, DBSCAN, hierarchiczna, PCA, autoenkodery, GMM |

#### Uczenie nadzorowane
Model minimalizuje różnicę między predykcją a etykietą. Zaleta: jasny cel i metryki. Wada: etykiety są kosztowne (czas ekspertów), mogą być zaszumione lub obciążone.

#### Uczenie nienadzorowane
Wykorzystuje nieopisane dane, które są tanie i liczne. Zastosowania: segmentacja klientów, kompresja, wizualizacja (t-SNE, UMAP), wstępne przetwarzanie, wykrywanie anomalii. Wada: brak jednoznacznej „poprawnej odpowiedzi”, wyniki zależą od hiperparametrów (np. liczby klastrów $k$).

#### Formy pośrednie
- **Semi-supervised** – mało etykiet + dużo danych bez etykiet.
- **Self-supervised** – etykiety generowane z samych danych (maskowanie słów w BERT, next-token w GPT, contrastive learning); podstawa współczesnych foundation models.
- **Weak supervision, active learning** – ograniczanie kosztu etykietowania.

#### Praktyka
Często łączy się oba podejścia: PCA/embeddingi (nienadzorowane) jako cechy dla klasyfikatora (nadzorowanego), albo klasteryzacja do eksploracji danych przed etykietowaniem.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
clf = LogisticRegression().fit(X, y)        # nadzorowane
km  = KMeans(n_clusters=3).fit(X)           # nienadzorowane
```

**Źródła:**
- [Supervised vs Unsupervised Learning (Outcome School)](https://outcomeschool.com/blog/supervised-vs-unsupervised-learning)
- [scikit-learn – Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)
- [scikit-learn – Unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html)
- [Wikipedia – Unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning)

---

<a id="q6"></a>
### 6. Czym jest uczenie ze wzmocnieniem (Reinforcement Learning)?

**Odpowiedź:**

**Reinforcement Learning (RL)** to paradygmat, w którym **agent** wchodzi w interakcję ze **środowiskiem**: w każdym kroku obserwuje stan $s_t$, wybiera akcję $a_t$, otrzymuje nagrodę $r_t$ i przechodzi do stanu $s_{t+1}$. Celem jest nauczenie **polityki** $\pi(a\mid s)$ maksymalizującej oczekiwaną skumulowaną (zdyskontowaną) nagrodę:

$$G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k+1},\quad \gamma\in[0,1)$$

#### Formalizm
Problem opisuje **Markov Decision Process (MDP)**: $(S, A, P, R, \gamma)$. Kluczowe funkcje:
- $V^\pi(s)$ – oczekiwany zwrot ze stanu $s$ przy polityce $\pi$,
- $Q^\pi(s,a)$ – oczekiwany zwrot po wykonaniu $a$ w $s$ i dalszym podążaniu za $\pi$.

Równanie Bellmana optymalności: $Q^*(s,a)=\mathbb{E}\big[r+\gamma\max_{a'}Q^*(s',a')\big]$.

#### Różnice względem uczenia nadzorowanego
- Brak etykiet „poprawnej akcji”; sygnał to opóźniona, rzadka nagroda.
- Dane zależą od polityki agenta (nie są i.i.d.).
- Problem przypisania zasługi (credit assignment) oraz kompromis **exploration vs exploitation**.

#### Rodziny algorytmów
- **Value-based**: Q-learning, DQN.
- **Policy-based**: REINFORCE, policy gradient.
- **Actor-critic**: A2C/A3C, PPO, SAC.
- **Model-based**: uczenie modelu środowiska (Dyna, MuZero).

#### Zastosowania
Gry (Atari, Go), robotyka, sterowanie, rekomendacje, optymalizacja zasobów oraz **RLHF** – dostrajanie LLM do preferencji ludzi.

#### Wyzwania
Niska efektywność próbkowania (sample efficiency), niestabilność treningu, projektowanie funkcji nagrody (reward hacking), bezpieczeństwo eksploracji, transfer z symulacji do rzeczywistości.

**Źródła:**
- [What is Reinforcement Learning? (Outcome School)](https://outcomeschool.com/blog/reinforcement-learning)
- [Sutton i Barto – Reinforcement Learning: An Introduction (2. wydanie)](http://incompleteideas.net/book/the-book-2nd.html)
- [Mnih i in., Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- [Schulman i in., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

---

<a id="q7"></a>
### 7. Czym jest bias?

**Odpowiedź:**

Termin „bias” ma w ML kilka znaczeń – warto je rozróżniać na rozmowie.

#### 1. Bias jako parametr neuronu / modelu (wyraz wolny)
W pojedynczym neuronie: $y=\sigma(w^\top x + b)$. **Bias $b$** przesuwa próg aktywacji, dzięki czemu funkcja nie musi przechodzić przez początek układu współrzędnych. Bez niego neuron z $x=0$ zawsze dawałby $\sigma(0)$. W regresji liniowej to intercept. Jest uczony razem z wagami; zwykle inicjalizowany zerami i zwykle nie jest regularyzowany.

#### 2. Bias statystyczny w kompromisie bias–variance
Błąd systematyczny estymatora: $\text{Bias}(\hat f(x)) = \mathbb{E}[\hat f(x)] - f(x)$. Oczekiwany błąd kwadratowy rozkłada się:

$$\mathbb{E}[(y-\hat f(x))^2]=\text{Bias}^2+\text{Variance}+\sigma^2_{\text{noise}}$$

- Wysoki bias = model zbyt prosty (**underfitting**), np. prosta dla danych nieliniowych.
- Wysoka wariancja = model zbyt wrażliwy na dane (**overfitting**).
- Złożoność modelu obniża bias, a podnosi wariancję – trzeba znaleźć równowagę.

#### 3. Bias indukcyjny (inductive bias)
Założenia wbudowane w model, umożliwiające generalizację: CNN zakłada lokalność i niezmienniczość translacyjną, RNN – sekwencyjność, Transformer – relacje przez attention.

#### 4. Bias w danych i sprawiedliwość (fairness)
Uprzedzenia w danych (selection bias, sampling bias, label bias) prowadzą do dyskryminujących modeli. Przeciwdziała się przez audyt danych, metryki sprawiedliwości (np. equalized odds), rebalansowanie, mitygację uprzedzeń i monitorowanie.

#### Jak diagnozować bias vs variance
Krzywe uczenia: wysoki błąd treningowy i walidacyjny → wysoki bias (większy model, więcej cech); niski trening, wysoki walidacja → wysoka wariancja (więcej danych, regularyzacja).

**Źródła:**
- [What is Bias? (Outcome School)](https://outcomeschool.com/blog/bias-in-artificial-neural-network)
- [Wikipedia – Bias–variance tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- [Deep Learning book, rozdz. 5.4: Estimators, Bias and Variance](https://www.deeplearningbook.org/contents/ml.html)
- [Mehrabi i in., A Survey on Bias and Fairness in Machine Learning](https://arxiv.org/abs/1908.09635)

---

<a id="q8"></a>
### 8. Jaka jest różnica między klasyfikacją a regresją?

**Odpowiedź:**

Oba zadania należą do uczenia nadzorowanego; różnią się typem zmiennej docelowej.

| | Klasyfikacja | Regresja |
|---|---|---|
| Wyjście | wartość dyskretna (klasa) | wartość ciągła |
| Przykłady | spam/nie spam, rozpoznanie cyfr, diagnoza | cena mieszkania, temperatura, popyt |
| Typowa loss | cross-entropy, hinge | MSE, MAE, Huber |
| Aktywacja wyjścia | sigmoid (binarna), softmax (wieloklasowa) | liniowa (brak) |
| Metryki | accuracy, precision, recall, F1, ROC-AUC | MAE, RMSE, $R^2$ |

#### Podtypy klasyfikacji
- **Binarna** (2 klasy), **wieloklasowa** (jedna z $K$), **wieloetykietowa** (multi-label; wiele klas jednocześnie).
- Modele zwracają zwykle prawdopodobieństwa; klasę wybiera się progiem (np. 0.5) – próg można dostosować do kosztów błędów.

#### Regresja
Model przewiduje liczbę: $\hat y = f_\theta(x)$. Wrażliwość na outliery zależy od loss (MSE karze je silniej niż MAE).

#### Uwagi praktyczne
- Nazwa „regresja logistyczna” jest myląca – to algorytm **klasyfikacji**.
- Regresję można zdyskretyzować (binning) do klasyfikacji, gdy liczy się przedział, ale tracimy informację.
- Wiele algorytmów ma oba warianty: drzewa decyzyjne, Random Forest, SVM/SVR, sieci neuronowe, gradient boosting.
- Niezbalansowanie klas to problem klasyfikacji; heteroskedastyczność i outliery to typowe problemy regresji.
- **Regresja porządkowa** (ordinal) leży pomiędzy – kategorie z porządkiem (np. oceny 1–5).

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
clf = RandomForestClassifier().fit(X, y_class)
reg = RandomForestRegressor().fit(X, y_value)
```

**Źródła:**
- [Classification vs Regression (Outcome School, LinkedIn)](https://www.linkedin.com/posts/outcomeschool_machinelearning-ai-activity-7372849348117835776-BWGL)
- [scikit-learn – Supervised learning](https://scikit-learn.org/stable/supervised_learning.html)
- [scikit-learn – Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Wikipedia – Statistical classification](https://en.wikipedia.org/wiki/Statistical_classification)

---

<a id="q9"></a>
### 9. Wyjaśnij overfitting i underfitting. Jak im zapobiegać?

**Odpowiedź:**

- **Underfitting** – model jest zbyt prosty, by uchwycić strukturę danych: wysoki błąd zarówno na zbiorze treningowym, jak i walidacyjnym (wysoki bias).
- **Overfitting** – model zapamiętuje szum i szczegóły zbioru treningowego: niski błąd treningowy, ale wyraźnie wyższy walidacyjny/testowy (wysoka wariancja). Słaba generalizacja.

#### Diagnoza
Krzywe uczenia (loss vs epoka lub vs rozmiar danych):
- train ≈ val, oba wysokie → underfitting,
- train niski, val rośnie/wysoki → overfitting,
- train ≈ val, oba niskie → dobry fit.

#### Jak zapobiegać overfittingowi
- **Więcej danych** i **data augmentation** (obrazy: obroty, kadrowanie; tekst: back-translation; dane tabelaryczne: SMOTE ostrożnie).
- **Regularyzacja**: L1/L2 (weight decay), dropout, early stopping.
- **Prostszy model**: mniej warstw/parametrów, przycinanie drzew (max_depth, min_samples_leaf).
- **Cross-validation** do wyboru hiperparametrów.
- **Redukcja cech** (feature selection, PCA).
- **Ensembling** (bagging, Random Forest) redukuje wariancję.
- **Batch normalization**, mniejszy learning rate, transfer learning (mniej parametrów do nauczenia od zera).

#### Jak zapobiegać underfittingowi
- Bardziej złożony model (głębsza sieć, więcej drzew, cechy wielomianowe).
- Lepsze cechy (feature engineering).
- Zmniejszenie regularyzacji.
- Dłuższy trening, lepszy optymalizator/learning rate.
- Sprawdzenie, czy dane niosą sygnał (czy zadanie w ogóle jest uczalne).

#### Pułapki
- **Data leakage** daje pozornie świetne wyniki walidacji.
- Wielokrotne strojenie na tym samym zbiorze walidacyjnym powoduje „overfitting do walidacji” – trzymaj osobny zbiór testowy.
- Zjawisko *double descent* pokazuje, że bardzo duże modele bywają wbrew klasycznej intuicji dobrze uogólniające.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
scores = cross_val_score(Ridge(alpha=1.0), X, y, cv=5)
```

**Źródła:**
- [Overfitting and Underfitting (Outcome School, LinkedIn)](https://www.linkedin.com/posts/outcomeschool_machinelearning-datascience-deeplearning-activity-7372237032145936384-gItd)
- [scikit-learn – Underfitting vs. Overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html)
- [Deep Learning book, rozdz. 7: Regularization for Deep Learning](https://www.deeplearningbook.org/contents/regularization.html)
- [Wikipedia – Overfitting](https://en.wikipedia.org/wiki/Overfitting)

---

<a id="q10"></a>
### 10. Czym są funkcje straty L1 i L2?

**Odpowiedź:**

Funkcje straty **L1** i **L2** mierzą błąd regresji jako normy różnicy między predykcją a wartością prawdziwą. (Nie mylić z regularyzacją L1/L2 – tu chodzi o samą loss, choć matematyka jest pokrewna.)

#### L1 Loss (MAE – Mean Absolute Error)
$$L_1=\frac{1}{N}\sum_{i=1}^N |y_i-\hat y_i|$$

- Kara rośnie liniowo z błędem → **odporna na outliery**.
- Estymator minimalizujący to **mediana** warunkowa.
- Gradient ma stałą wartość ($\pm1$), nieciągły w 0 → wolniejsza zbieżność w pobliżu optimum (stały krok), wymaga subgradientu.

#### L2 Loss (MSE – Mean Squared Error)
$$L_2=\frac{1}{N}\sum_{i=1}^N (y_i-\hat y_i)^2$$

- Kara rośnie kwadratowo → duże błędy są silnie karane, **wrażliwa na outliery**.
- Estymator minimalizujący to **średnia** warunkowa; odpowiada estymacji największej wiarygodności przy gaussowskim szumie.
- Gradient $2(\hat y-y)$ maleje przy zbliżaniu do optimum – gładka, zbieżna optymalizacja.

#### Porównanie
| | L1 (MAE) | L2 (MSE) |
|---|---|---|
| Outliery | odporna | wrażliwa |
| Gładkość | nieróżniczkowalna w 0 | gładka |
| Estymator | mediana | średnia |
| Interpretacja | w jednostkach zmiennej | w jednostkach kwadratowych (RMSE wraca do jednostek) |

#### Kompromis: Huber Loss
Kwadratowa dla małych błędów ($|e|\le\delta$), liniowa dla dużych – gładka i odporna. Alternatywy: log-cosh, quantile loss.

#### Wybór
Jeśli w danych są outliery, które są błędami pomiaru → L1/Huber. Jeśli duże błędy są szczególnie kosztowne w biznesie → L2.

```python
import torch.nn as nn
l1 = nn.L1Loss(); l2 = nn.MSELoss(); huber = nn.HuberLoss(delta=1.0)
```

**Źródła:**
- [What Are L1 and L2 Loss Functions? (Outcome School)](https://outcomeschool.com/blog/l1-and-l2-loss-functions)
- [PyTorch – Loss functions (torch.nn)](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [Wikipedia – Huber loss](https://en.wikipedia.org/wiki/Huber_loss)
- [Wikipedia – Mean absolute error](https://en.wikipedia.org/wiki/Mean_absolute_error)

---

<a id="q11"></a>
### 11. Czym jest regularyzacja? Wyjaśnij regularyzację L1 (Lasso) i L2 (Ridge).

**Odpowiedź:**

**Regularyzacja** to zestaw technik ograniczających złożoność modelu, aby zmniejszyć overfitting. Najczęściej dodaje się do funkcji straty karę za wielkość wag:

$$J(w)=\underbrace{L(w)}_{\text{dane}}+\lambda\,\Omega(w)$$

gdzie $\lambda\ge0$ steruje siłą regularyzacji (większe $\lambda$ = prostszy model, większy bias, mniejsza wariancja).

#### L2 (Ridge, weight decay)
$$\Omega(w)=\|w\|_2^2=\sum_j w_j^2$$

- Ściąga wagi w stronę zera, ale rzadko dokładnie do zera.
- Gradient: $2\lambda w$ – aktualizacja „zanik wag”.
- Rozwiązanie w regresji liniowej: $w=(X^\top X+\lambda I)^{-1}X^\top y$ – poprawia uwarunkowanie przy współliniowości cech.
- Interpretacja bayesowska: prior gaussowski na wagach.

#### L1 (Lasso)
$$\Omega(w)=\|w\|_1=\sum_j |w_j|$$

- Prowadzi do **rzadkich** wag (wiele dokładnie zerowych) → automatyczny **wybór cech**.
- Geometria: romb (kula L1) ma wierzchołki na osiach, więc rozwiązanie często trafia w zera.
- Interpretacja bayesowska: prior Laplace'a.
- Przy silnie skorelowanych cechach wybiera zwykle jedną z nich arbitralnie.

#### Elastic Net
$\lambda_1\|w\|_1+\lambda_2\|w\|_2^2$ – łączy rzadkość i stabilność przy skorelowanych cechach.

#### Inne formy regularyzacji
Dropout, early stopping, data augmentation, batch norm (efekt uboczny), ograniczanie głębokości drzew, label smoothing.

#### Praktyka
- **Skaluj cechy** (standaryzacja) przed regularyzacją – kara zależy od skali wag.
- Nie regularyzuj biasu (zwykle).
- $\lambda$ dobieraj cross-validation (`RidgeCV`, `LassoCV`).
- W Adam preferuj **AdamW** (rozdzielony weight decay), bo L2 wpleciona w gradient działa inaczej.

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
Ridge(alpha=1.0).fit(X, y); Lasso(alpha=0.1).fit(X, y)
```

**Źródła:**
- [Regularization In Machine Learning (Outcome School)](https://outcomeschool.com/blog/regularization-in-machine-learning)
- [scikit-learn – Linear Models (Ridge, Lasso, Elastic-Net)](https://scikit-learn.org/stable/modules/linear_model.html)
- [Deep Learning book, rozdz. 7.1: Parameter Norm Penalties](https://www.deeplearningbook.org/contents/regularization.html)
- [Loshchilov i Hutter, Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)

---

<a id="q12"></a>
### 12. Czym są funkcje straty (loss function) i funkcje kosztu (cost function)? Jaka jest kluczowa różnica?

**Odpowiedź:**

- **Loss function** $\ell(\hat y_i,y_i)$ – miara błędu dla **pojedynczej** próbki.
- **Cost function** $J(\theta)$ – agregacja strat na **całym zbiorze** (lub batchu), często z dodanym członem regularyzacji:

$$J(\theta)=\frac{1}{N}\sum_{i=1}^N \ell\big(f_\theta(x_i),y_i\big)+\lambda\,\Omega(\theta)$$

W praktyce (i w wielu bibliotekach) terminy są używane wymiennie; w rygorystycznym ujęciu (np. kurs Andrew Nga) loss dotyczy jednego przykładu, a cost – całego zbioru. Trzeci termin, **objective function**, jest najogólniejszy: to to, co optymalizujemy (może być maksymalizowane, np. likelihood).

#### Przykłady loss
- Regresja: MSE, MAE, Huber.
- Klasyfikacja: binary cross-entropy, categorical cross-entropy, hinge (SVM), focal loss.
- Rankingi/embeddingi: triplet loss, contrastive/InfoNCE.
- Generacja: negative log-likelihood, KL divergence.

#### Wymagania od dobrej loss
Różniczkowalna (lub subdifferentiable) dla optymalizacji gradientowej, dobrze skorelowana z metryką biznesową, odpowiednio skalowana i odporna numerycznie.

#### Loss a metryka
Loss optymalizujemy podczas treningu (musi być różniczkowalna), a **metryka** (accuracy, F1, AUC) służy do ewaluacji i bywa nieróżniczkowalna. Dlatego używamy surogatów – np. cross-entropy zamiast 0–1 loss.

#### Pułapki
Niezgodność loss z metryką, niezbalansowane klasy (rozważ wagi lub focal loss), zły dobór skali w zadaniach wielozadaniowych.

```python
import torch.nn.functional as F
loss = F.cross_entropy(logits, targets)   # średnia strata w batchu = cost
```

**Źródła:**
- [Loss functions and Cost functions in Machine Learning (Amit Shekhar, X)](https://x.com/amitiitbhu/status/1925471830640664925)
- [Wikipedia – Loss function](https://en.wikipedia.org/wiki/Loss_function)
- [Deep Learning book, rozdz. 6.2: Gradient-Based Learning](https://www.deeplearningbook.org/contents/mlp.html)
- [PyTorch – Loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions)

---

<a id="q13"></a>
### 13. Czym jest dropout?

**Odpowiedź:**

**Dropout** to technika regularyzacji sieci neuronowych: podczas treningu każdy neuron (jego wyjście) jest losowo „wyłączany” (zerowany) z prawdopodobieństwem $p$, niezależnie w każdym kroku. Uniemożliwia to nadmierną ko-adaptację neuronów – sieć nie może polegać na pojedynczych cechach i uczy się redundantnych, bardziej odpornych reprezentacji.

#### Mechanizm
Trening: $\tilde h = m \odot h$, gdzie $m_j\sim\text{Bernoulli}(1-p)$.
Inferencja: dropout jest wyłączony. Aby zachować skalę oczekiwanej aktywacji, stosuje się **inverted dropout**: podczas treningu dzieli się przez $(1-p)$, więc przy inferencji nic nie trzeba skalować.

#### Interpretacja
- Każdy krok trenuje inną, „przerzedzoną” podsieć; wynik przypomina uśrednianie (ensemble) wykładniczo wielu podsieci.
- Można ją traktować jako przybliżenie wnioskowania bayesowskiego (Monte Carlo dropout do szacowania niepewności: dropout włączony także przy predykcji, wiele przebiegów).

#### Wskazówki praktyczne
- Typowe $p$: 0.1–0.5 (0.5 dla warstw gęstych, mniej dla warstw konwolucyjnych i Transformerów, ok. 0.1).
- Stosuj po aktywacji w warstwach gęstych; nie w warstwie wyjściowej.
- Wymaga zwykle dłuższego treningu (szum spowalnia zbieżność).
- Zawsze `model.eval()` przy ewaluacji.
- Z batch normalization bywa mało skuteczny lub kolidujący; w CNN często zastępuje się go innymi technikami (DropBlock, augmentacje).
- Warianty: DropConnect, spatial dropout, DropPath (stochastic depth).

```python
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(), nn.Dropout(p=0.5),
    nn.Linear(256, 10)
)
model.train()   # dropout aktywny
model.eval()    # dropout wyłączony
```

**Źródła:**
- [Dropout in Neural Networks (Outcome School)](https://outcomeschool.com/blog/dropout-in-neural-networks)
- [Srivastava i in., Dropout: A Simple Way to Prevent Neural Networks from Overfitting (JMLR)](https://jmlr.org/papers/v15/srivastava14a.html)
- [Gal i Ghahramani, Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142)
- [PyTorch – torch.nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)

---

<a id="q14"></a>
### 14. Czym jest perceptron?

**Odpowiedź:**

**Perceptron** (Rosenblatt, 1958) to najprostszy model sztucznego neuronu i liniowy klasyfikator binarny. Oblicza ważoną sumę wejść plus bias i przepuszcza ją przez funkcję progową:

$$\hat y=\begin{cases}1 & w^\top x+b>0\\ 0 & \text{w przeciwnym razie}\end{cases}$$

#### Reguła uczenia
Dla każdej próbki $(x,y)$, $y\in\{0,1\}$:

$$w \leftarrow w+\eta\,(y-\hat y)\,x,\qquad b\leftarrow b+\eta\,(y-\hat y)$$

Wagi zmieniają się tylko przy błędnej klasyfikacji.

#### Właściwości
- **Twierdzenie o zbieżności**: jeśli dane są liniowo separowalne, algorytm zbiega w skończonej liczbie kroków.
- Granica decyzyjna to hiperpłaszczyzna $w^\top x+b=0$.
- Dla danych nieseparowalnych liniowo nie zbiega (oscyluje).

#### Ograniczenia
Nie rozwiąże problemu **XOR** (Minsky i Papert, 1969), bo wymaga nieliniowej granicy. To m.in. doprowadziło do pierwszej „zimy AI”. Rozwiązaniem są sieci wielowarstwowe (MLP) z nieliniowymi aktywacjami, uczone backpropagation.

#### Perceptron vs regresja logistyczna
Perceptron używa progu i uczy się tylko na błędach; regresja logistyczna używa sigmoidy, zwraca prawdopodobieństwa i minimalizuje log-loss (gładka, różniczkowalna).

#### Znaczenie
Podstawowy „klocek” sieci neuronowych; wariant *averaged/voted perceptron* i *kernel perceptron* jest używany do dziś w prostych zadaniach online.

```python
from sklearn.linear_model import Perceptron
clf = Perceptron(eta0=1.0, max_iter=1000).fit(X, y)
```

**Źródła:**
- [Perceptron in Machine Learning (Amit Shekhar, X)](https://x.com/amitiitbhu/status/2011100039545078038)
- [Wikipedia – Perceptron](https://en.wikipedia.org/wiki/Perceptron)
- [scikit-learn – Perceptron](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html)
- [Stanford CS229 – materiały do kursu (perceptron, klasyfikatory liniowe)](https://cs229.stanford.edu/)

---

<a id="q15"></a>
### 15. Wyjaśnij wielowarstwowy perceptron (MLP – Multilayer Perceptron).

**Odpowiedź:**

**MLP** to sieć neuronowa typu feed-forward złożona z warstwy wejściowej, jednej lub więcej **warstw ukrytych** oraz warstwy wyjściowej. Każdy neuron jest połączony ze wszystkimi neuronami poprzedniej warstwy (warstwy w pełni połączone, fully connected/dense).

#### Obliczenia
Dla warstwy $l$:

$$h^{(l)}=\sigma\big(W^{(l)}h^{(l-1)}+b^{(l)}\big)$$

gdzie $\sigma$ jest nieliniową funkcją aktywacji (ReLU, GELU, tanh, sigmoid). Warstwa wyjściowa: liniowa (regresja), sigmoid (binarna), softmax (wieloklasowa).

#### Dlaczego nieliniowość jest kluczowa
Bez niej złożenie warstw liniowych jest nadal przekształceniem liniowym. Z nieliniowościami MLP rozwiązuje XOR i inne problemy nieseparowalne liniowo.

#### Uniwersalne twierdzenie aproksymacji
MLP z jedną warstwą ukrytą i wystarczającą liczbą neuronów może aproksymować dowolną ciągłą funkcję na zbiorze zwartym (z rozsądnymi aktywacjami). Nie mówi jednak nic o tym, jak łatwo taką sieć nauczyć; głębokość zwykle daje większą efektywność parametrów.

#### Uczenie
1. **Forward pass** – obliczenie predykcji i loss.
2. **Backpropagation** – gradienty przez regułę łańcuchową.
3. **Optymalizator** (SGD, Adam) aktualizuje wagi.

#### Zalety i wady
+ Uniwersalny aproksymator, działa na danych tabelarycznych i jako blok w większych architekturach (np. feed-forward w Transformerze).
− Wiele parametrów, brak wbudowanego inductive bias dla obrazów/sekwencji (CNN/RNN/Transformer lepsze), skłonność do overfittingu, wrażliwość na skalę cech i inicjalizację.

#### Praktyka
Standaryzacja wejść, inicjalizacja He/Xavier, ReLU, regularyzacja (dropout, weight decay), batch/layer norm.

```python
import torch.nn as nn
mlp = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 3)          # logity 3 klas
)
```

**Źródła:**
- [Understanding Multilayer Perceptron (MLP) Neural Network (Amit Shekhar, LinkedIn)](https://www.linkedin.com/posts/amit-shekhar-iitbhu_machinelearning-deeplearning-neuralnetwork-activity-7331528906673401857-bpqL)
- [Deep Learning book, rozdz. 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)
- [scikit-learn – Neural network models (supervised)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [Wikipedia – Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)

---

<a id="q16"></a>
### 16. Czym jest cross-entropy?

**Odpowiedź:**

**Cross-entropy (entropia krzyżowa)** mierzy, ile bitów (lub nat) średnio potrzeba do zakodowania zdarzeń z rozkładu prawdziwego $p$ przy użyciu kodu zoptymalizowanego dla rozkładu $q$ (predykcji modelu):

$$H(p,q)=-\sum_x p(x)\log q(x)=H(p)+D_{KL}(p\,\|\,q)$$

Ponieważ $H(p)$ jest stałą, minimalizowanie cross-entropy równa się minimalizowaniu dywergencji KL między rozkładem prawdziwym a modelem, a także maksymalizacji **log-likelihood**.

#### W klasyfikacji
- **Binarna (log loss)**: $L=-\big[y\log\hat p+(1-y)\log(1-\hat p)\big]$
- **Wieloklasowa (one-hot $y$)**: $L=-\sum_{k}y_k\log\hat p_k=-\log\hat p_{c}$, gdzie $c$ to prawdziwa klasa.

Przykład: prawdziwa klasa 1, predykcja $(0.7,0.2,0.1)$ → $L=-\ln 0.7\approx0.357$. Predykcja $0.1$ dla prawdziwej klasy daje $-\ln0.1\approx2.30$ – pewne błędy są karane bardzo mocno.

#### Dlaczego lepsza niż MSE dla klasyfikacji
Z softmax/sigmoid gradient cross-entropy względem logitów wynosi $\hat p-y$ – nie zanika, gdy model jest pewnie błędny, podczas gdy MSE z sigmoidą ma gradient tłumiony przez pochodną sigmoidy (wolniejsza nauka). Jest też uzasadniona probabilistycznie (MLE).

#### Praktyka
- W PyTorch `CrossEntropyLoss` przyjmuje **surowe logity** (łączy log-softmax i NLL); nie stosuj softmax przed nią.
- Niezbalansowane klasy: parametr `weight` lub focal loss.
- Label smoothing zmniejsza nadmierną pewność.
- Perplexity w LLM to $\exp(\text{cross-entropy})$.

```python
import torch, torch.nn.functional as F
logits = torch.tensor([[2.0, 1.0, 0.1]])
loss = F.cross_entropy(logits, torch.tensor([0]))
```

**Źródła:**
- [Math Behind Cross-Entropy Loss (Outcome School)](https://outcomeschool.com/blog/math-behind-cross-entropy-loss)
- [Wikipedia – Cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy)
- [Deep Learning book, rozdz. 3.13: Information Theory](https://www.deeplearningbook.org/contents/prob.html)
- [PyTorch – torch.nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

---

<a id="q17"></a>
### 17. Czym są logity (logits)?

**Odpowiedź:**

**Logity** to surowe, nieznormalizowane wyjścia ostatniej warstwy liniowej klasyfikatora – przed zastosowaniem sigmoidy lub softmaxu. Mogą przyjmować dowolne wartości rzeczywiste (ujemne, większe od 1), nie są prawdopodobieństwami.

#### Pochodzenie nazwy
W statystyce logit to $\text{logit}(p)=\ln\frac{p}{1-p}$ (log-odds), funkcja odwrotna do sigmoidy: $p=\sigma(z)=\frac{1}{1+e^{-z}}$. W regresji logistycznej $z=w^\top x+b$ jest właśnie logarytmem szans. W ML rozszerzono termin na wektor wyjść przed softmaxem.

#### Przekształcenie w prawdopodobieństwa
- Binarnie: $p=\sigma(z)$.
- Wieloklasowo: $p_i=\text{softmax}(z)_i=e^{z_i}/\sum_je^{z_j}$.

Przykład: logity $(2.0,1.0,0.1)$ → prawdopodobieństwa $\approx(0.66,0.24,0.10)$.

#### Dlaczego pracuje się na logitach
- **Stabilność numeryczna**: `BCEWithLogitsLoss` i `CrossEntropyLoss` liczą log-sum-exp w stabilny sposób, unikając przepełnień i $\log(0)$.
- Softmax zachowuje kolejność, więc argmax z logitów = argmax z prawdopodobieństw – do samej klasyfikacji softmax nie jest potrzebny.
- Logity zachowują informację o „pewności” w skali, której nie widać po nasyceniu prawdopodobieństw.

#### Zastosowania
- Temperature scaling (kalibracja): $\text{softmax}(z/T)$.
- Sampling w LLM: top-k/top-p/temperature działają na logitach.
- Knowledge distillation: dopasowanie logitów nauczyciela i ucznia.
- Wykrywanie OOD (np. energy score).

```python
import torch
logits = model(x)                       # (batch, num_classes), surowe wartości
probs  = torch.softmax(logits, dim=-1)  # prawdopodobieństwa
loss   = torch.nn.functional.cross_entropy(logits, y)  # przyjmuje logity
```

**Źródła:**
- [Understanding Logits in Machine Learning (Amit Shekhar, X)](https://x.com/amitiitbhu/status/1927927814923207146)
- [Wikipedia – Logit](https://en.wikipedia.org/wiki/Logit)
- [PyTorch – torch.nn.BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
- [Guo i in., On Calibration of Modern Neural Networks (temperature scaling)](https://arxiv.org/abs/1706.04599)

---

<a id="q18"></a>
### 18. Wyjaśnij cross-validation. Dlaczego się ją stosuje?

**Odpowiedź:**

**Cross-validation (walidacja krzyżowa)** to metoda oceny zdolności generalizacji modelu poprzez wielokrotny podział danych na część treningową i walidacyjną, tak aby każda obserwacja posłużyła do walidacji. Wynik to średnia (i odchylenie standardowe) metryki z kilku podziałów.

#### K-fold
1. Podziel dane na $k$ równych części (foldów; zwykle $k=5$ lub $10$).
2. Dla $i=1..k$: trenuj na $k-1$ foldach, waliduj na $i$-tym.
3. Uśrednij wyniki: $\text{CV}=\frac1k\sum_i \text{score}_i$.

#### Dlaczego się ją stosuje
- Bardziej **wiarygodna i mniej wariancyjna ocena** niż pojedynczy podział train/val, zwłaszcza dla małych zbiorów.
- Pełniejsze wykorzystanie danych.
- **Dobór hiperparametrów i modeli** (GridSearchCV, RandomizedSearchCV, Optuna).
- Wykrywanie overfittingu i niestabilności modelu (duży rozrzut między foldami).

#### Warianty
- **Stratified k-fold** – zachowuje proporcje klas (niezbalansowana klasyfikacja).
- **Leave-One-Out (LOO)** – $k=N$, niski bias, wysoki koszt i wariancja.
- **Group k-fold** – próbki tej samej grupy (np. pacjenta) trafiają do jednego foldu, by uniknąć wycieku.
- **TimeSeriesSplit / walk-forward** – dla szeregów czasowych; trening zawsze przed walidacją (nie tasować!).
- **Nested CV** – zewnętrzna pętla do oceny, wewnętrzna do strojenia hiperparametrów; daje nieobciążoną ocenę.
- **Repeated k-fold** – powtórzenia z różnymi losowaniami.

#### Pułapki
- **Data leakage**: skalowanie, imputacja, selekcja cech muszą być wykonywane *wewnątrz* foldów (użyj `Pipeline`).
- Zbiór testowy nadal trzymaj osobno – CV służy do wyboru modelu, nie do końcowej oceny.
- Wysoki koszt dla dużych modeli (deep learning zwykle używa pojedynczego podziału).
- Bias–variance wyboru $k$: małe $k$ – większy bias (mniej danych treningowych), duże $k$ – większa wariancja i koszt.

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1")
```

**Źródła:**
- [Understanding Cross-Validation in Machine Learning (Amit Shekhar, X)](https://x.com/amitiitbhu/status/1939929240084128137)
- [scikit-learn – Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Wikipedia – Cross-validation (statistics)](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
- [Cawley i Talbot, On Over-fitting in Model Selection and Subsequent Selection Bias (JMLR)](https://jmlr.org/papers/v11/cawley10a.html)

---

<a id="q19"></a>
### 19. Czym są precision, recall i F1-score?

**Odpowiedź:**

Metryki te opisują jakość klasyfikatora binarnego w oparciu o **macierz pomyłek**:

| | Predykcja: pozytywna | Predykcja: negatywna |
|---|---|---|
| Rzeczywista: pozytywna | TP | FN |
| Rzeczywista: negatywna | FP | TN |

#### Definicje
- **Precision** (precyzja) – jaka część przewidzianych pozytywów jest naprawdę pozytywna: $\text{Precision}=\frac{TP}{TP+FP}$.
- **Recall** (czułość, sensitivity, TPR) – jaką część rzeczywistych pozytywów model wykrył: $\text{Recall}=\frac{TP}{TP+FN}$.
- **F1-score** – średnia harmoniczna: $F_1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}$. Średnia harmoniczna karze niezrównoważone wartości (np. $P=1$, $R=0.1$ daje $F_1\approx0.18$, nie 0.55).
- Uogólnienie: $F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}$ – $\beta>1$ faworyzuje recall, $\beta<1$ precision.

#### Przykład
Wykryto 100 spamów, z czego 80 prawdziwych; w zbiorze jest 200 spamów. $P=80/100=0.8$, $R=80/200=0.4$, $F_1=2\cdot0.8\cdot0.4/1.2\approx0.53$.

#### Kompromis precision–recall
Zmiana progu decyzyjnego: niższy próg → wyższy recall, niższa precision; wyższy próg – odwrotnie. Wybór zależy od kosztu błędów:
- **Recall ważniejszy**: diagnostyka medyczna, wykrywanie oszustw, bezpieczeństwo (FN kosztuje więcej).
- **Precision ważniejsza**: filtry spamu (nie chcemy blokować ważnej poczty), rekomendacje z kosztowną interwencją.

#### Dlaczego nie samo accuracy
Przy niezbalansowanych klasach (1% pozytywnych) model „zawsze negatywny” ma 99% accuracy, ale recall 0. Precision/recall/F1 i **PR-AUC** są wtedy bardziej informatywne (ROC-AUC bywa zbyt optymistyczny).

#### Uśrednianie dla wielu klas
- **macro** – średnia po klasach (równe wagi), **micro** – globalne TP/FP/FN, **weighted** – ważona liczebnością klas.

```python
from sklearn.metrics import precision_recall_fscore_support, classification_report
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
print(classification_report(y_true, y_pred))
```

**Źródła:**
- [Precision vs Recall (Outcome School)](https://outcomeschool.com/blog/precision-vs-recall)
- [scikit-learn – Precision, recall and F-measures](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
- [Wikipedia – Precision and recall](https://en.wikipedia.org/wiki/Precision_and_recall)
- [Saito i Rehmsmeier, The Precision-Recall Plot Is More Informative than the ROC Plot (PLOS ONE)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)

---

<a id="q20"></a>
### 20. Czym jest wykrywanie anomalii (anomaly detection)?

**Odpowiedź:**

**Anomaly detection** (wykrywanie anomalii, outlierów, novelty detection) to identyfikacja obserwacji, które istotnie odbiegają od oczekiwanego wzorca „normalnego” zachowania. Zastosowania: wykrywanie oszustw (fraud), intruzów w sieci, awarii maszyn (predictive maintenance), wad produkcyjnych, anomalii w metrykach systemowych i w danych medycznych.

#### Typy anomalii
- **Punktowe** – pojedyncza nietypowa wartość.
- **Kontekstowe** – normalne globalnie, nietypowe w kontekście (np. 30°C w styczniu).
- **Zbiorowe** – sekwencja punktów nietypowa łącznie.

#### Ustawienia nauki
- **Nadzorowane** – etykiety normalne/anomalia; rzadko dostępne i skrajnie niezbalansowane.
- **Półnadzorowane (one-class/novelty)** – trening tylko na danych normalnych.
- **Nienadzorowane** – bez etykiet, zakłada się, że anomalii jest mało.

#### Metody
- **Statystyczne**: z-score, IQR, test Grubbsa, estymacja gęstości (GMM).
- **Odległościowe/gęstościowe**: kNN, **LOF** (Local Outlier Factor), DBSCAN.
- **Drzewiaste**: **Isolation Forest** – anomalie izoluje się losowymi podziałami w krótszych ścieżkach.
- **Granicowe**: One-Class SVM.
- **Rekonstrukcyjne**: autoenkodery/VAE – duży błąd rekonstrukcji oznacza anomalię; PCA.
- **Szeregi czasowe**: modele prognozy (ARIMA, Prophet, LSTM) i analiza reszt; rozkład sezonowy (STL).

#### Ewaluacja i praktyka
- Ze względu na niezbalansowanie używa się precision/recall, PR-AUC, precision@k, nie accuracy.
- Trudność: zdefiniowanie „normalności”, dryf danych, koszt fałszywych alarmów (alert fatigue), brak etykiet.
- Dobór progu na podstawie kosztu błędów; ludzie w pętli (human-in-the-loop) do weryfikacji.

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.01, random_state=0).fit(X)
labels = iso.predict(X)          # -1 = anomalia, 1 = norma
scores = iso.score_samples(X)
```

**Źródła:**
- [scikit-learn – Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Chandola, Banerjee, Kumar – Anomaly Detection: A Survey (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/1541880.1541882)
- [Wikipedia – Anomaly detection](https://en.wikipedia.org/wiki/Anomaly_detection)
- [Wikipedia – Isolation forest](https://en.wikipedia.org/wiki/Isolation_forest)

---

<a id="q21"></a>
### 21. Jaka jest różnica między metodami policy-based i value-based?

**Odpowiedź:**

W RL istnieją dwa główne sposoby znajdowania optymalnej strategii.

#### Value-based
Uczymy się **funkcji wartości** ($V(s)$ lub $Q(s,a)$), a politykę wyprowadzamy pośrednio, zwykle zachłannie:

$$\pi(s)=\arg\max_a Q(s,a)$$

Przykłady: Q-learning, SARSA, DQN (i warianty Double/Dueling/Rainbow).
- Zalety: zwykle lepsza efektywność próbkowania (off-policy, replay buffer), prosta idea.
- Wady: trudność z **ciągłymi przestrzeniami akcji** (max po $a$ bywa nietrywialny), polityki deterministyczne (eksplorację trzeba dodać, np. $\varepsilon$-greedy), możliwa niestabilność z aproksymacją funkcji (deadly triad: bootstrapping + off-policy + function approximation).

#### Policy-based
Parametryzujemy **politykę** $\pi_\theta(a\mid s)$ i optymalizujemy ją bezpośrednio gradientem oczekiwanego zwrotu (policy gradient theorem):

$$\nabla_\theta J(\theta)=\mathbb{E}_{\pi_\theta}\big[\nabla_\theta\log\pi_\theta(a\mid s)\,A(s,a)\big]$$

Przykłady: REINFORCE, TRPO, PPO.
- Zalety: naturalne dla ciągłych akcji, polityki stochastyczne (pomocne w środowiskach częściowo obserwowalnych i grach), lepsza zbieżność do lokalnego optimum, płynna zmiana polityki.
- Wady: wysoka wariancja gradientu, niższa efektywność próbkowania (często on-policy), zbieżność do optimum lokalnego.

#### Actor-critic
Łączy oba światy: **actor** (polityka) i **critic** (funkcja wartości redukująca wariancję przez baseline/advantage). Przykłady: A2C, PPO, SAC, DDPG, TD3.

| | Value-based | Policy-based |
|---|---|---|
| Uczony obiekt | $Q$ / $V$ | $\pi_\theta$ |
| Akcje ciągłe | trudne | naturalne |
| Polityka | pośrednia, deterministyczna | bezpośrednia, stochastyczna |
| Wariancja | niższa | wyższa |
| Przykłady | DQN | REINFORCE, PPO |

**Źródła:**
- [Sutton i Barto – Reinforcement Learning: An Introduction, rozdz. 13: Policy Gradient Methods](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Spinning Up – Kinds of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)
- [Mnih i in., Human-level control through deep reinforcement learning (DQN)](https://www.nature.com/articles/nature14236)
- [Schulman i in., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

---

<a id="q22"></a>
### 22. Czym jest Q-Learning?

**Odpowiedź:**

**Q-learning** (Watkins, 1989) to algorytm RL typu **value-based, off-policy, model-free**, który uczy się funkcji akcji–wartości $Q(s,a)$: oczekiwanego zdyskontowanego zwrotu po wykonaniu akcji $a$ w stanie $s$ i dalszym postępowaniu optymalnie.

#### Reguła aktualizacji
Po przejściu $(s,a,r,s')$:

$$Q(s,a)\leftarrow Q(s,a)+\alpha\Big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\Big]$$

- $\alpha$ – learning rate, $\gamma$ – współczynnik dyskontowania.
- Człon w nawiasie to **TD error** (temporal difference).
- **Off-policy**: cel używa $\max_{a'}$, niezależnie od akcji faktycznie wybranej przez politykę eksplorującą – uczy się polityki optymalnej mimo zachowań eksploracyjnych. (SARSA jest on-policy: używa faktycznie wybranej $a'$.)

#### Algorytm (tabularny)
1. Zainicjuj $Q(s,a)$ (np. zerami).
2. W każdym kroku wybierz $a$ polityką $\varepsilon$-greedy, wykonaj, zaobserwuj $r,s'$.
3. Zaktualizuj $Q$ regułą powyżej; $s\leftarrow s'$.
4. Powtarzaj do zbieżności. Przy skończonej liczbie stanów i akcji, odpowiednim zmniejszaniu $\alpha$ i odwiedzaniu wszystkich par nieskończenie często, $Q\to Q^*$.

#### Ograniczenia i rozszerzenia
- Tabela $Q$ nie skaluje się na duże przestrzenie stanów → **Deep Q-Network (DQN)**: sieć neuronowa aproksymuje $Q$, z **experience replay** i **target network** dla stabilności.
- Przeszacowanie wartości (maximization bias) → **Double DQN**.
- Akcje ciągłe → DDPG/SAC (actor-critic).

```python
import numpy as np
def q_update(Q, s, a, r, s_next, alpha=0.1, gamma=0.99):
    td_target = r + gamma * np.max(Q[s_next])
    Q[s, a] += alpha * (td_target - Q[s, a])
```

**Źródła:**
- [Wikipedia – Q-learning](https://en.wikipedia.org/wiki/Q-learning)
- [Sutton i Barto – RL: An Introduction, rozdz. 6.5: Q-learning](http://incompleteideas.net/book/the-book-2nd.html)
- [Mnih i in., Playing Atari with Deep Reinforcement Learning (DQN)](https://arxiv.org/abs/1312.5602)
- [van Hasselt i in., Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)

---

<a id="q23"></a>
### 23. Wyjaśnij koncepcję exploration vs exploitation.

**Odpowiedź:**

To fundamentalny dylemat w RL i problemach decyzyjnych:

- **Exploitation (eksploatacja)** – wybieranie akcji, która według dotychczasowej wiedzy daje największą nagrodę.
- **Exploration (eksploracja)** – próbowanie mniej znanych akcji, aby zdobyć informację, która może ujawnić lepsze opcje.

Sama eksploatacja grozi utknięciem w suboptymalnej strategii; sama eksploracja marnuje nagrody. Trzeba je zbalansować, zwykle z większą eksploracją na początku i mniejszą później.

#### Klasyczny przykład: multi-armed bandit
Kilka automatów o nieznanych rozkładach wypłat. Który ciągnąć? Miarą jakości jest **regret** – strata względem zawsze wybieranej najlepszej ręki.

#### Strategie
- **$\varepsilon$-greedy** – z prawdopodobieństwem $\varepsilon$ losowa akcja, inaczej najlepsza; często $\varepsilon$ maleje w czasie (annealing).
- **Softmax / Boltzmann** – akcje wybierane z prawdopodobieństwem $\propto e^{Q(a)/\tau}$.
- **UCB (Upper Confidence Bound)** – optymizm w obliczu niepewności: $a=\arg\max\big[\hat\mu_a+c\sqrt{\ln t/n_a}\big]$.
- **Thompson sampling** – losowanie z rozkładu a posteriori wartości akcji i wybór maksimum; bayesowskie i bardzo skuteczne w praktyce.
- **Optimistic initialization** – wysoka wartość początkowa zachęca do próbowania.
- **Entropy regularization** (SAC, PPO) – bonus za stochastyczną politykę.
- **Intrinsic motivation / curiosity** (RND, ICM) – nagroda za nowość, ważna przy rzadkich nagrodach.

#### Zastosowania poza RL
Testy A/B i adaptacyjne (bandity kontekstowe) w rekomendacjach i reklamach, dobór hiperparametrów (Bayesian optimization), sampling w LLM (temperature).

#### Praktyka
Eksploracja ma koszt (w produkcji – realne straty biznesowe), więc stosuje się ograniczenia bezpieczeństwa, offline evaluation i off-policy learning.

```python
import numpy as np
def eps_greedy(Q, s, eps):
    return np.random.randint(Q.shape[1]) if np.random.rand() < eps else int(np.argmax(Q[s]))
```

**Źródła:**
- [Sutton i Barto – RL: An Introduction, rozdz. 2: Multi-armed Bandits](http://incompleteideas.net/book/the-book-2nd.html)
- [Wikipedia – Exploration–exploitation dilemma](https://en.wikipedia.org/wiki/Exploration%E2%80%93exploitation_dilemma)
- [Auer, Cesa-Bianchi, Fischer – Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023/A:1013689704352)
- [Russo i in., A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038)

---

<a id="q24"></a>
### 24. Wyjaśnij przekleństwo wymiarowości (curse of dimensionality) i jak sobie z nim radzić.

**Odpowiedź:**

**Curse of dimensionality** (Bellman) to zbiór zjawisk, w których wraz ze wzrostem liczby wymiarów $d$ przestrzeń cech „rozrzedza się” tak szybko, że algorytmy stają się nieefektywne lub tracą jakość.

#### Główne przejawy
- **Wykładniczy wzrost objętości**: aby pokryć przestrzeń o stałej gęstości, liczba próbek rośnie wykładniczo z $d$ ($\sim k^d$). Dane stają się rzadkie (sparse).
- **Koncentracja odległości**: w wysokich wymiarach odległości między punktami stają się niemal równe – stosunek $\frac{d_{max}-d_{min}}{d_{min}}\to0$; pojęcie „najbliższego sąsiada” traci sens (problem dla kNN, klasteryzacji).
- **Geometria kuli**: objętość hiperkuli jednostkowej względem opisanego hipersześcianu dąży do zera; prawie cała masa leży w cienkiej warstwie przy powierzchni.
- **Overfitting**: więcej cech niż próbek → model łatwo dopasowuje szum (zasada: liczba potrzebnych danych rośnie z wymiarem).
- **Koszt obliczeń i pamięci** rośnie.

#### Jak sobie radzić
1. **Redukcja wymiaru – ekstrakcja cech**: PCA, SVD, LDA, autoenkodery, t-SNE/UMAP (głównie wizualizacja), random projections (lemat Johnsona–Lindenstraussa).
2. **Selekcja cech**: filtry (korelacja, mutual information, wariancja), wrappery (RFE), metody wbudowane (L1/Lasso, ważność cech z drzew).
3. **Regularyzacja** (L1/L2, dropout) i prostsze modele.
4. **Więcej danych** oraz augmentacja.
5. **Wiedza domenowa** i inżynieria cech – zmniejszenie liczby cech.
6. **Uczenie reprezentacji / embeddingi** – gęste, niskowymiarowe reprezentacje.
7. **Odpowiednie metryki i algorytmy**: cosine distance, przybliżone wyszukiwanie sąsiadów (ANN: HNSW, LSH) zamiast dokładnego kNN; modele niewrażliwe na wymiar (drzewa, boosting).
8. **Założenie o rozmaitości (manifold hypothesis)** – dane realne leżą na niskowymiarowej rozmaitości, dlatego głębokie sieci mimo wielu wymiarów działają.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)      # zachowaj 95% wariancji
X_low = pca.fit_transform(X_scaled)
```

**Źródła:**
- [Wikipedia – Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)
- [Deep Learning book, rozdz. 5.11: Challenges Motivating Deep Learning](https://www.deeplearningbook.org/contents/ml.html)
- [scikit-learn – Decomposing signals in components (PCA)](https://scikit-learn.org/stable/modules/decomposition.html)
- [Hastie, Tibshirani, Friedman – The Elements of Statistical Learning (rozdz. 2.5: Local Methods in High Dimensions)](https://hastie.su.domains/ElemStatLearn/)

---

<a id="q25"></a>
### 25. Wyjaśnij Local Loss, Focal Loss i Gradient Blending w kontekście Multi-Task Learning.

**Odpowiedź:**

**Multi-Task Learning (MTL)** trenuje jeden model (zwykle ze współdzielonym „trzonem” i osobnymi głowami) na wielu zadaniach jednocześnie. Główne trudności: zadania mają różne skale strat, różne tempo uczenia i mogą powodować **konflikty gradientów** (negative transfer). Trzeba więc dobrze zbalansować wkład zadań.

#### Local Loss (strata lokalna)
Termin nie jest standaryzowany; w kontekście MTL zwykle oznacza **stratę pojedynczego zadania/głowy** (lub stratę liczoną na poziomie lokalnym, np. piksela/regionu, w przeciwieństwie do straty globalnej łączonej). Całkowita strata to kombinacja strat lokalnych:

$$L_{total}=\sum_t w_t\,L_t$$

Wagi $w_t$ można ustawić ręcznie, uczyć (np. uncertainty weighting Kendalla i in.: $\sum_t \frac{1}{2\sigma_t^2}L_t+\log\sigma_t$) lub adaptować dynamicznie (GradNorm, dynamic weight average). Na rozmowie warto zaznaczyć, że termin bywa używany różnie, i wyjaśnić przyjętą definicję.

#### Focal Loss
Zaproponowana w RetinaNet do klasyfikacji z ekstremalnym niezbalansowaniem (np. detekcja obiektów, gdzie tło dominuje). Modyfikuje cross-entropy, zmniejszając wagę łatwych przykładów:

$$FL(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t)$$

gdzie $p_t$ to prawdopodobieństwo prawdziwej klasy, $\gamma\ge0$ (np. 2) to parametr skupienia. Dla $\gamma=0$ otrzymujemy cross-entropy. Łatwy przykład ($p_t=0.9$) ma czynnik $(0.1)^2=0.01$ – jego wkład maleje ~100 razy, więc gradient koncentruje się na trudnych przykładach. W MTL stosuje się ją w zadaniach klasyfikacyjnych niezbalansowanych oraz jako sposób „automatycznego” równoważenia trudności.

#### Gradient Blending
Pochodzi z pracy o multimodalnych sieciach (Wang i in., *What Makes Training Multi-Modal Classification Networks Hard?*). Problem: modalności (wideo, audio) overfitują w różnym tempie, więc wspólny trening bywa gorszy niż najlepsza pojedyncza modalność. **Gradient Blending (G-Blend)** oblicza optymalne wagi strat dla poszczególnych modalności/gałęzi na podstawie tego, jak *overfitting-to-generalization ratio* zmienia się w czasie, i miksuje gradienty tak, by każdy kanał wnosił użyteczny sygnał, a nie tylko zapamiętywał dane treningowe.

#### Powiązane techniki równoważenia gradientów
- **GradNorm** – normalizuje tempo uczenia zadań przez dopasowanie norm gradientów.
- **PCGrad** – projekcja gradientów w konflikcie na płaszczyznę normalną do drugiego.
- **Uncertainty weighting**, **MGDA** (Pareto).

#### Praktyka
Monitoruj metryki każdego zadania osobno, normalizuj skale strat, testuj baseline single-task, uważaj na dominację jednego zadania.

```python
loss = sum(w[t] * criterion[t](out[t], y[t]) for t in tasks)   # L_total
```

**Źródła:**
- [Lin i in., Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [Wang, Tran, Feiszli, What Makes Training Multi-Modal Classification Networks Hard? (Gradient Blending)](https://arxiv.org/abs/1905.12681)
- [Chen i in., GradNorm: Gradient Normalization for Adaptive Loss Balancing](https://arxiv.org/abs/1711.02257)
- [Kendall, Gal, Cipolla, Multi-Task Learning Using Uncertainty to Weigh Losses](https://arxiv.org/abs/1705.07115)
- [Yu i in., Gradient Surgery for Multi-Task Learning (PCGrad)](https://arxiv.org/abs/2001.06782)

---

<a id="q26"></a>
### 26. Wyjaśnij Contrastive Learning.

**Odpowiedź:**

**Contrastive learning** to metoda uczenia reprezentacji (najczęściej self-supervised), w której model uczy się przestrzeni embeddingów tak, aby **pary pozytywne** (podobne) były blisko, a **pary negatywne** (niepodobne) daleko od siebie – bez potrzeby ręcznych etykiet klas.

#### Idea
Dla próbki (kotwicy) $x$ tworzymy przykład pozytywny $x^+$ (np. inna augmentacja tego samego obrazu, sąsiednie zdanie, tekst opisujący obraz) i negatywy $x^-_j$ (inne próbki z batcha). Encoder $f$ uczymy tak, by $f(x)\approx f(x^+)$ i $f(x)\not\approx f(x^-)$.

#### Funkcje straty
- **Contrastive loss (Hadsell i in.)** dla par: przyciąga pozytywy, odpycha negatywy poza margines $m$.
- **Triplet loss**: $\max(0,\;d(a,p)-d(a,n)+m)$.
- **InfoNCE / NT-Xent** (SimCLR, CPC, CLIP):

$$\mathcal{L}=-\log\frac{\exp(\text{sim}(z,z^+)/\tau)}{\sum_{k=0}^{K}\exp(\text{sim}(z,z_k)/\tau)}$$

gdzie sim to zwykle cosine similarity, a $\tau$ – temperatura. To w istocie cross-entropy rozpoznająca pozytyw wśród negatywów; związana z dolnym ograniczeniem informacji wzajemnej.

#### Przykłady
- **SimCLR, MoCo** – obrazy: dwie losowe augmentacje jednego obrazu = para pozytywna.
- **CLIP** – pary obraz–tekst z internetu; kontrastowy trening dwóch enkoderów, umożliwia zero-shot classification.
- **Sentence embeddings (SimCSE, sentence-transformers)** – wyszukiwanie semantyczne, RAG.
- **Non-contrastive**: BYOL, SimSiam, Barlow Twins – bez negatywów (unikają kolapsu innymi mechanizmami).

#### Praktyka i pułapki
- Duże batche/kolejki pamięci (MoCo) dostarczają więcej negatywów.
- **Hard negatives** poprawiają jakość, ale **false negatives** (fałszywie uznane za niepodobne) szkodzą.
- Dobór augmentacji jest kluczowy (decyduje, jakie niezmienniki model się nauczy).
- Zbyt niska temperatura → niestabilność; zbyt wysoka → słabe rozróżnianie.

```python
import torch, torch.nn.functional as F
def info_nce(q, k, tau=0.07):           # q,k: (B,d) znormalizowane; k[i] to pozytyw dla q[i]
    logits = q @ k.T / tau
    return F.cross_entropy(logits, torch.arange(len(q)))
```

**Źródła:**
- [Contrastive Learning (Outcome School)](https://outcomeschool.com/blog/contrastive-learning)
- [Chen i in., A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709)
- [van den Oord i in., Representation Learning with Contrastive Predictive Coding (InfoNCE)](https://arxiv.org/abs/1807.03748)
- [Radford i in., Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)

---

<a id="q27"></a>
### 27. Czym jest Generative AI?

**Odpowiedź:**

**Generative AI (GenAI)** to klasa modeli uczących się rozkładu danych $p(x)$ (lub warunkowego $p(x\mid c)$), aby **generować nowe, realistyczne treści**: tekst, obrazy, audio, wideo, kod, struktury molekularne. W odróżnieniu od modeli **dyskryminatywnych**, które uczą $p(y\mid x)$ (klasyfikują lub przewidują), modele generatywne tworzą nowe próbki podobne do danych treningowych.

#### Główne rodziny modeli
- **Autoregresyjne (Transformer, GPT)** – faktoryzacja $p(x)=\prod_t p(x_t\mid x_{<t})$, generacja token po tokenie; podstawa LLM.
- **VAE (Variational Autoencoder)** – enkoder do przestrzeni latentnej + dekoder; trening przez ELBO.
- **GAN (Generative Adversarial Network)** – generator kontra dyskryminator w grze minimax; ostre obrazy, ale niestabilny trening i mode collapse.
- **Modele dyfuzyjne (diffusion)** – uczą się odszumiania krok po kroku; Stable Diffusion, DALL·E, modele wideo; dziś dominują w obrazach.
- **Flow-based / energy-based** – dokładne wiarygodności lub energie.

#### Foundation models i alignment
Wielkie modele pre-trenowane na ogromnych danych (self-supervised), następnie dostosowywane: instruction tuning, **RLHF/DPO**, fine-tuning (LoRA), RAG i tool use. Prompting umożliwia zero-/few-shot.

#### Zastosowania
Asystenci konwersacyjni, generowanie kodu, streszczanie, tłumaczenie, tworzenie obrazów i grafiki, synteza mowy, generowanie danych syntetycznych, odkrywanie leków, agenci.

#### Ograniczenia i ryzyka
- **Halucynacje** (płynne, ale fałszywe treści),
- uprzedzenia i toksyczność, wyciek danych, kwestie praw autorskich,
- deepfake i dezinformacja,
- koszt obliczeń, opóźnienia, trudność ewaluacji (brak jednej metryki – używa się FID, BLEU/ROUGE ograniczenie, LLM-as-judge, oceny ludzkiej),
- bezpieczeństwo (prompt injection, jailbreak).

#### Dobre praktyki
Grounding (RAG), guardrails, ewaluacja na własnych zestawach testowych, human-in-the-loop, monitoring i filtracja treści.

**Źródła:**
- [What is Generative AI? (Outcome School)](https://outcomeschool.com/blog/what-is-generative-ai)
- [Goodfellow i in., Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Kingma i Welling, Auto-Encoding Variational Bayes (VAE)](https://arxiv.org/abs/1312.6114)
- [Ho, Jain, Abbeel, Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Vaswani i in., Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## Algorytmy

<a id="q28"></a>
### 28. Jak działa algorytm drzewa decyzyjnego (Decision Tree)?

**Odpowiedź:**

**Drzewo decyzyjne** to nieparametryczny model, który rekurencyjnie dzieli przestrzeń cech na prostokątne regiony za pomocą prostych reguł „cecha ≤ próg”. Węzły wewnętrzne zawierają test, gałęzie – wyniki testu, a liście – predykcję (klasa lub średnia wartość).

#### Budowa (algorytm zachłanny, top-down)
1. Zacznij w korzeniu z całym zbiorem.
2. Dla każdej cechy i możliwego progu oblicz jakość podziału.
3. Wybierz podział maksymalizujący spadek zanieczyszczenia (impurity).
4. Powtórz rekurencyjnie na potomkach, aż do spełnienia warunku stopu (max_depth, min_samples_leaf, brak zysku, czysty węzeł).
5. Liść zwraca klasę większościową (klasyfikacja) lub średnią (regresja).

#### Miary zanieczyszczenia
- **Gini**: $G=1-\sum_k p_k^2$.
- **Entropia**: $H=-\sum_k p_k\log_2 p_k$; **information gain** $=H(\text{rodzic})-\sum_i\frac{n_i}{n}H(\text{dziecko}_i)$.
- **Regresja**: redukcja wariancji / MSE.

**Przykład**: węzeł 10 próbek, 5 klasy A i 5 klasy B: $G=1-(0.25+0.25)=0.5$ (maksimum dla 2 klas). Podział na (5A,0B) i (0A,5B) daje $G=0$, więc zysk 0.5.

#### Zalety
Interpretowalność (można narysować reguły), brak potrzeby skalowania cech, obsługa danych mieszanych i nieliniowości, uwzględnianie interakcji cech, szybka predykcja.

#### Wady
- **Overfitting** – nieprzycięte drzewo zapamiętuje dane; wysoka wariancja, niestabilność (mała zmiana danych → inne drzewo).
- Granice prostopadłe do osi – słabo aproksymują ukośne granice.
- Zachłanność nie gwarantuje optimum globalnego.
- Regresja: predykcje schodkowe, brak ekstrapolacji.
- Skłonność do cech o wielu wartościach.

#### Kontrola złożoności
Pre-pruning (`max_depth`, `min_samples_split`, `min_samples_leaf`), post-pruning (cost-complexity pruning, `ccp_alpha`).

#### Algorytmy
ID3, C4.5, **CART** (binarne podziały, Gini; używany w scikit-learn).

```python
from sklearn.tree import DecisionTreeClassifier, export_text
tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(X, y)
print(export_text(tree))
```

**Źródła:**
- [scikit-learn – Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Wikipedia – Decision tree learning](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [Hastie, Tibshirani, Friedman – The Elements of Statistical Learning, rozdz. 9.2: Tree-Based Methods](https://hastie.su.domains/ElemStatLearn/)
- [Stanford CS229 – Decision Trees (notatki)](https://cs229.stanford.edu/)

---

<a id="q29"></a>
### 29. Wyjaśnij, jak drzewa decyzyjne dokonują podziałów i obsługują cechy kategoryczne.

**Odpowiedź:**

#### Wybór podziału (split)
W każdym węźle algorytm przeszukuje zachłannie wszystkie kandydackie podziały i wybiera najlepszy pod względem kryterium.

**Cechy numeryczne:**
1. Sortuje wartości cechy.
2. Kandydatami na progi są punkty pośrodku sąsiednich, różnych wartości.
3. Dla każdego progu $t$ oblicza zanieczyszczenie ważone: 

$$\Delta = I(\text{rodzic})-\frac{n_L}{n}I(L)-\frac{n_R}{n}I(R)$$

4. Wybiera cechę i próg z maksymalnym $\Delta$. Złożoność dla jednego węzła to $O(d\,n\log n)$ (sortowanie), a przy histogramowych implementacjach (LightGBM, XGBoost hist) sprowadza się do przeszukiwania kubełków.

**Kryteria:** Gini, entropia/information gain (klasyfikacja), MSE/MAE (regresja). Information gain ma skłonność do cech o wielu wartościach – C4.5 stosuje więc **gain ratio**.

#### Cechy kategoryczne
Sposób zależy od implementacji:

- **Kodowanie one-hot** i traktowanie jak numeryczne binarne – proste, ale przy wysokiej kardynalności tworzy wiele rzadkich cech i „płytkie” drzewa.
- **Kodowanie porządkowe (ordinal/label)** – tylko gdy istnieje naturalny porządek; inaczej wprowadza sztuczne relacje.
- **Podziały na podzbiorach kategorii** – teoretycznie $2^{K-1}-1$ podziałów binarnych, ale dla klasyfikacji binarnej i regresji istnieje sztuczka: **posortuj kategorie według średniej wartości celu** i szukaj progu w tej kolejności – daje optymalny podział w $O(K\log K)$ (Fisher, CART). Wspierane natywnie m.in. w LightGBM, CatBoost, `HistGradientBoosting` w scikit-learn (`categorical_features`).
- **Target encoding** (średnia celu per kategoria) – uwaga na wyciek celu; stosuj z out-of-fold/wygładzaniem (CatBoost używa *ordered target statistics*).
- **Multiway split (ID3/C4.5)** – jedna gałąź na kategorię; podatne na overfitting.
- Rzadkie kategorie łącz w „inne”.

#### Brakujące wartości
CART: *surrogate splits*; XGBoost/LightGBM uczą domyślny kierunek dla braków; scikit-learn (≥1.3) wspiera NaN w drzewach.

#### Pułapki
- One-hot dla kardynalności rzędu tysięcy → słabe wyniki; lepsze natywne kategorie lub encoding.
- Zbyt drobne progi prowadzą do overfittingu – kontroluj `min_samples_leaf`.

```python
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(categorical_features="from_dtype").fit(X_df, y)
```

**Źródła:**
- [scikit-learn – Decision Trees: mathematical formulation i criteria](https://scikit-learn.org/stable/modules/tree.html#mathematical-formulation)
- [scikit-learn – Histogram-based Gradient Boosting (obsługa cech kategorycznych)](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
- [LightGBM – Features: Optimal Split for Categorical Features](https://lightgbm.readthedocs.io/en/latest/Features.html)
- [Prokhorenkova i in., CatBoost: unbiased boosting with categorical features](https://arxiv.org/abs/1706.09516)

---

<a id="q30"></a>
### 30. Jak działa Random Forest? Jak ulepsza drzewa decyzyjne i jak redukuje wariancję?

**Odpowiedź:**

**Random Forest** (Breiman, 2001) to zespół (ensemble) wielu drzew decyzyjnych trenowanych na losowych wariantach danych, których predykcje są agregowane: głosowanie większościowe (klasyfikacja) lub średnia (regresja).

#### Algorytm
Dla $b=1..B$:
1. **Bootstrap** – wylosuj ze zwracaniem próbę o rozmiarze $N$ (bagging); ok. 63% unikalnych próbek trafia do próby.
2. Zbuduj drzewo (zwykle głębokie, bez przycinania), a w **każdym węźle** rozważaj tylko losowy podzbiór $m$ cech (zwykle $m\approx\sqrt d$ dla klasyfikacji, $d/3$ dla regresji).
3. Predykcja: $\hat f(x)=\frac1B\sum_b T_b(x)$ (lub głosowanie).

#### Dlaczego redukuje wariancję
Pojedyncze głębokie drzewo ma niski bias, ale wysoką wariancję. Dla $B$ zmiennych o wariancji $\sigma^2$ i średniej korelacji $\rho$ wariancja średniej wynosi:

$$\text{Var}=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$$

- Uśrednianie zmniejsza drugi człon przy $B\to\infty$.
- **Losowanie cech** obniża korelację $\rho$ między drzewami – bez tego dominujące cechy powodowałyby podobne drzewa i ograniczały zysk z baggingu.
- Bias pozostaje zbliżony do bias pojedynczego drzewa.

#### Ulepszenia względem pojedynczego drzewa
Znacznie lepsza generalizacja, odporność na szum i outliery, mniejsza wrażliwość na hiperparametry, mniej przycinania.

#### Dodatkowe zalety
- **OOB (out-of-bag) error** – ocena generalizacji bez osobnego zbioru walidacyjnego (próbki nie użyte w danym drzewie).
- **Feature importance** (impurity-based, permutation importance) – uwaga: impurity-based faworyzuje cechy o wysokiej kardynalności.
- Łatwa paralelizacja, mała potrzeba preprocessingu.

#### Wady
Mniejsza interpretowalność, większy model i wolniejsza inferencja, słaba ekstrapolacja, zwykle ustępuje dobrze dostrojonemu gradient boostingowi na danych tabelarycznych.

#### Kluczowe hiperparametry
`n_estimators`, `max_features`, `max_depth`, `min_samples_leaf`, `max_samples`, `class_weight`, `oob_score`.

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", oob_score=True, n_jobs=-1, random_state=0).fit(X, y)
print(rf.oob_score_)
```

**Źródła:**
- [Wikipedia – Random forest](https://en.wikipedia.org/wiki/Random_forest)
- [scikit-learn – Ensembles: Forests of randomized trees](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Breiman – Random Forests (Machine Learning, 2001)](https://link.springer.com/article/10.1023/A:1010933404324)
- [Hastie, Tibshirani, Friedman – ESL, rozdz. 15: Random Forests](https://hastie.su.domains/ElemStatLearn/)

---

<a id="q31"></a>
### 31. Wyjaśnij metody zespołowe (Ensemble Methods). Dlaczego są tak skuteczne?

**Odpowiedź:**

**Ensemble learning** łączy predykcje wielu modeli bazowych (weak/base learners) w jedną, zwykle dokładniejszą i stabilniejszą. Idea „mądrości tłumu”: jeśli modele popełniają **różne, niezupełnie skorelowane błędy**, ich agregacja te błędy częściowo znosi.

#### Dlaczego działają
- **Redukcja wariancji** (bagging): uśrednianie skorelowanych z $\rho<1$ estymatorów obniża wariancję: $\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$.
- **Redukcja biasu** (boosting): sekwencyjne dodawanie modeli poprawiających błędy poprzednich zmniejsza błąd systematyczny.
- **Rozszerzenie klasy hipotez**: kombinacja (np. stacking) reprezentuje funkcje, których pojedynczy model nie odwzoruje.
- Warunek: modele muszą być **różnorodne** i każdy lepszy niż losowy. Ekstremalny przykład: $B$ niezależnych klasyfikatorów o accuracy 60% daje przy głosowaniu większościowym accuracy dążące do 100% wraz z $B$ (przy pełnej niezależności; w praktyce korelacja ogranicza zysk).

#### Główne rodziny
1. **Bagging** (Bootstrap Aggregating) – równoległe modele na próbach bootstrap; Random Forest, ExtraTrees.
2. **Boosting** – sekwencyjne, każdy model skupia się na błędach poprzednich; AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost.
3. **Stacking** – meta-model uczy się łączyć predykcje modeli bazowych (trenowane na predykcjach out-of-fold, aby uniknąć wycieku).
4. **Voting / averaging / blending** – proste (hard/soft voting) lub ważone uśrednianie różnorodnych modeli.

#### Techniki tworzenia różnorodności
Różne algorytmy, losowe podzbiory danych i cech, różne inicjalizacje/architektury, różne hiperparametry, snapshot ensembles i checkpoint averaging w sieciach.

#### Wady
- Większy koszt treningu i inferencji, większe zużycie pamięci,
- trudniejsza interpretowalność i wdrożenie,
- malejące korzyści przy skorelowanych modelach; ryzyko wycieku w stackingu.
- W konkursach (Kaggle) ensemble są standardem; w produkcji często kompromis (distillation ensemblu do jednego modelu).

```python
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
ens = VotingClassifier([("lr", LogisticRegression(max_iter=1000)),
                        ("rf", RandomForestClassifier()),
                        ("gb", GradientBoostingClassifier())], voting="soft").fit(X, y)
```

**Źródła:**
- [scikit-learn – Ensembles: Gradient boosting, forests, bagging, voting, stacking](https://scikit-learn.org/stable/modules/ensemble.html)
- [Wikipedia – Ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning)
- [Dietterich – Ensemble Methods in Machine Learning](https://link.springer.com/chapter/10.1007/3-540-45014-9_1)
- [Wolpert – Stacked Generalization](https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231)

---

<a id="q32"></a>
### 32. Jaka jest różnica między baggingiem a boostingiem?

**Odpowiedź:**

Oba to metody zespołowe łączące wiele słabych modeli, ale różnią się sposobem ich trenowania i tym, jaki błąd redukują.

| Aspekt | Bagging | Boosting |
|---|---|---|
| Trening | **równoległy**, niezależne modele | **sekwencyjny**, każdy model zależy od poprzednich |
| Dane | próby bootstrap | wszystkie dane z ważeniem błędów (AdaBoost) lub residuami/gradientami (GBM) |
| Modele bazowe | złożone, niski bias, wysoka wariancja (głębokie drzewa) | proste, wysoki bias (płytkie drzewa, stumps) |
| Redukuje | głównie **wariancję** | głównie **bias** (i częściowo wariancję) |
| Agregacja | średnia/głosowanie równe wagi | ważona suma modeli |
| Ryzyko overfittingu | niskie | wyższe (wymaga regularyzacji, early stopping) |
| Odporność na szum/outliery | duża | mniejsza (błędne etykiety są wzmacniane) |
| Równoległość | łatwa | ograniczona (choć drzewa budowane równolegle wewnętrznie) |
| Przykłady | Random Forest, Bagged Trees | AdaBoost, GBM, XGBoost, LightGBM, CatBoost |

#### Bagging
Każdy model uczony na losowej próbie bootstrap; predykcja to uśrednienie. Uśrednianie stabilizuje niestabilne modele (drzewa). Daje darmowy OOB error.

#### Boosting
- **AdaBoost**: po każdej rundzie zwiększa wagi błędnie sklasyfikowanych próbek; model dostaje wagę zależną od jego dokładności.
- **Gradient boosting**: kolejny model dopasowuje **ujemny gradient** funkcji straty (dla MSE – residua): $F_m=F_{m-1}+\eta\,h_m$.

#### Kiedy co
- Model bazowy overfituje (wysoka wariancja), dane zaszumione → bagging/Random Forest.
- Model bazowy underfituje, potrzebna maksymalna dokładność na danych tabelarycznych → boosting (po starannym strojeniu zwykle wygrywa).
- Bagging jest prostszy i mniej wrażliwy na hiperparametry; boosting wymaga strojenia learning rate, liczby drzew i głębokości.

```python
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
bag = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, oob_score=True)
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200, learning_rate=0.5)
```

**Źródła:**
- [scikit-learn – Bagging meta-estimator i AdaBoost](https://scikit-learn.org/stable/modules/ensemble.html)
- [Breiman – Bagging Predictors (Machine Learning, 1996)](https://link.springer.com/article/10.1007/BF00058655)
- [Freund i Schapire – A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://www.sciencedirect.com/science/article/pii/S002200009791504X)
- [Wikipedia – Bootstrap aggregating](https://en.wikipedia.org/wiki/Bootstrap_aggregating)

---

<a id="q33"></a>
### 33. Czym jest Gradient Boosting? Jak działa XGBoost?

**Odpowiedź:**

#### Gradient Boosting
**Gradient boosting** (Friedman, 2001) buduje model addytywny stopniowo, traktując boosting jako **gradient descent w przestrzeni funkcji**:

$$F_m(x)=F_{m-1}(x)+\eta\,h_m(x)$$

W każdej iteracji:
1. Oblicz **pseudo-residua** – ujemny gradient loss względem bieżących predykcji: $r_i=-\frac{\partial L(y_i,F(x_i))}{\partial F(x_i)}\Big|_{F=F_{m-1}}$ (dla MSE to zwykłe residua $y_i-F_{m-1}(x_i)$).
2. Dopasuj słabe drzewo $h_m$ (zwykle płytkie, CART) do $r_i$.
3. Dodaj je do modelu ze współczynnikiem $\eta$ (shrinkage, learning rate).

Dzięki uogólnieniu na dowolną różniczkowalną loss (log-loss, MSE, quantile, ranking) metoda pasuje do klasyfikacji, regresji i rankingu.

#### XGBoost (eXtreme Gradient Boosting)
Wydajna, regularyzowana implementacja gradient boostingu (Chen i Guestrin, 2016). Kluczowe elementy:

- **Regularyzowany cel**: $\mathcal{L}=\sum_i l(y_i,\hat y_i)+\sum_k\Omega(f_k)$, gdzie $\Omega(f)=\gamma T+\tfrac12\lambda\sum_j w_j^2$ ($T$ – liczba liści, $w_j$ – wagi liści).
- **Rozwinięcie Taylora drugiego rzędu**: używa gradientów $g_i$ i hesjanów $h_i$. Optymalna waga liścia: $w_j^*=-\frac{\sum_{i\in I_j}g_i}{\sum_{i\in I_j}h_i+\lambda}$.
- **Zysk podziału**: 

$$\text{Gain}=\tfrac12\Big[\frac{G_L^2}{H_L+\lambda}+\frac{G_R^2}{H_R+\lambda}-\frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\Big]-\gamma$$

  podział wykonuje się tylko, gdy $\text{Gain}>0$ (wbudowane przycinanie przez $\gamma$).
- **Wyszukiwanie podziałów**: dokładne (greedy) lub przybliżone/histogramowe (`tree_method="hist"`), sketch kwantyli.
- **Sparsity-aware split finding**: uczy domyślnego kierunku dla brakujących wartości.
- **Optymalizacje systemowe**: równoległe przeszukiwanie cech, struktura blokowa w pamięci, cache-awareness, obsługa out-of-core, GPU.
- **Regularyzacja stochastyczna**: `subsample`, `colsample_bytree` – losowanie wierszy i kolumn (jak w Random Forest).
- **Early stopping** na zbiorze walidacyjnym.

#### Zalety i wady
+ Bardzo wysoka jakość na danych tabelarycznych, radzi sobie z brakami, nieliniowościami i interakcjami; szybki.
− Wrażliwy na hiperparametry, ryzyko overfittingu przy zbyt wielu/głębokich drzewach, słaba ekstrapolacja, mniej interpretowalny (SHAP pomaga).

```python
import xgboost as xgb
model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

**Źródła:**
- [Chen i Guestrin, XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [XGBoost – Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html)
- [Friedman – Greedy Function Approximation: A Gradient Boosting Machine](https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full)
- [scikit-learn – Gradient Tree Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)

---

<a id="q34"></a>
### 34. Jakie są kluczowe hiperparametry XGBoost?

**Odpowiedź:**

Hiperparametry XGBoost można pogrupować według wpływu na złożoność modelu i regularyzację.

#### Kontrola tempa uczenia i liczby drzew
- **`n_estimators`** (`num_boost_round`) – liczba drzew. Ustaw dużą wartość i użyj early stopping.
- **`learning_rate`** (`eta`, domyślnie 0.3) – skala wkładu każdego drzewa; mniejsza (0.01–0.1) = lepsza generalizacja, ale więcej drzew. Trade-off z `n_estimators`.

#### Złożoność drzewa
- **`max_depth`** (domyślnie 6) – głębokość; typowo 3–10. Większa = wyższa zdolność, ryzyko overfittingu.
- **`min_child_weight`** – minimalna suma hesjanów w liściu; większe = bardziej konserwatywny model.
- **`gamma`** (`min_split_loss`) – minimalny zysk wymagany do podziału; większe = mocniejsze przycinanie.
- **`max_leaves`, `grow_policy`** – przy `lossguide` (jak LightGBM) rośnie po liściach.

#### Losowość (redukcja wariancji)
- **`subsample`** – ułamek wierszy na drzewo (0.5–1).
- **`colsample_bytree`, `colsample_bylevel`, `colsample_bynode`** – ułamek cech.

#### Regularyzacja
- **`reg_lambda`** (L2, domyślnie 1) i **`reg_alpha`** (L1, domyślnie 0) na wagi liści.

#### Niezbalansowanie i cel
- **`scale_pos_weight`** ≈ liczba negatywów / liczba pozytywów; alternatywnie wagi próbek.
- **`objective`** (`binary:logistic`, `multi:softprob`, `reg:squarederror`, `rank:pairwise`...) i **`eval_metric`** (`logloss`, `auc`, `rmse`...).
- **`max_delta_step`** – stabilizuje przy skrajnie niezbalansowanych danych.

#### Wydajność
- **`tree_method`** (`hist`, `approx`, `exact`, `gpu_hist`/`device="cuda"`), `n_jobs`, `max_bin`.

#### Strategia strojenia
1. Ustal `learning_rate` (np. 0.05–0.1), włącz early stopping i wyznacz liczbę drzew.
2. Stroj `max_depth` i `min_child_weight`.
3. Stroj `gamma`, potem `subsample` i `colsample_bytree`.
4. Dodaj regularyzację (`reg_lambda`, `reg_alpha`).
5. Obniż `learning_rate` i zwiększ liczbę drzew dla finalnego modelu.
Użyj random search lub Bayesian optimization (Optuna), zawsze z cross-validation.

#### Diagnoza
- Overfitting (train ≪ val): zmniejsz `max_depth`, zwiększ `min_child_weight`, `gamma`, `reg_lambda`, obniż `subsample`/`colsample`.
- Underfitting: zwiększ głębokość i liczbę drzew, zmniejsz regularyzację.

```python
import xgboost as xgb
params = dict(objective="binary:logistic", eval_metric="auc", eta=0.05, max_depth=6,
              min_child_weight=3, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
bst = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dval, "val")], early_stopping_rounds=50)
```

**Źródła:**
- [XGBoost – Parameters (oficjalna dokumentacja)](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [XGBoost – Notes on Parameter Tuning](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html)
- [Chen i Guestrin, XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [Optuna – dokumentacja](https://optuna.readthedocs.io/)

---

<a id="q35"></a>
### 35. Wyjaśnij Gradient Boosting i jego zalety w porównaniu z Random Forests.

**Odpowiedź:**

**Gradient Boosting** buduje model addytywny sekwencyjnie: każde kolejne drzewo (słaby uczeń, zwykle płytkie drzewo) uczy się poprawiać błędy dotychczasowego zespołu. Model ma postać

$$F_M(x) = F_0(x) + \eta \sum_{m=1}^{M} h_m(x)$$

gdzie $\eta$ to learning rate (shrinkage), a $h_m$ dopasowujemy do **ujemnego gradientu funkcji straty** względem bieżących predykcji (tzw. pseudo-residuals):

$$r_i^{(m)} = -\left.\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F=F_{m-1}}$$

Dla MSE pseudo-residuals to po prostu reszty $y - F(x)$, dla log-loss - $y - p$. Dzięki temu można stosować dowolną różniczkowalną stratę (regresja, klasyfikacja, ranking).

**Random Forest** to bagging: wiele głębokich, niezależnych drzew na próbkach bootstrap z losowym podzbiorem cech, uśrednianych w celu redukcji **wariancji**. Boosting redukuje głównie **bias** (i częściowo wariancję).

| Cecha | Random Forest | Gradient Boosting |
|---|---|---|
| Trening | równoległy, niezależne drzewa | sekwencyjny |
| Drzewa | głębokie, niskie obciążenie | płytkie (głębokość 3-8) |
| Główny cel | redukcja wariancji | redukcja biasu |
| Podatność na overfitting | niska | wyższa, wymaga regularyzacji |
| Strojenie | proste, odporny na domyślne parametry | wrażliwy (learning rate, liczba drzew, głębokość) |

**Zalety GB względem RF:** zwykle wyższa dokładność na danych tabelarycznych, elastyczność funkcji straty, mniejsze modele przy tej samej jakości, natywna obsługa brakujących wartości w nowoczesnych implementacjach (XGBoost, LightGBM, CatBoost).

**Wady:** wolniejszy trening (sekwencyjność), łatwiej o overfitting, więcej hiperparametrów, wrażliwość na szum w etykietach.

**Praktyka:** mały learning rate (0.01-0.1) + duża liczba drzew + early stopping na zbiorze walidacyjnym; subsampling wierszy i cech (stochastic GB), regularyzacja L1/L2 liści, `min_child_weight`. Implementacje: XGBoost, LightGBM (histogramy, leaf-wise), CatBoost (ordered boosting, kategorie).

```python
from sklearn.ensemble import HistGradientBoostingClassifier
clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=500,
                                     early_stopping=True, random_state=0)
clf.fit(X_train, y_train)
```

**Źródła:**
- [scikit-learn: Ensemble methods (Gradient Boosting, Random Forests)](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost: A Scalable Tree Boosting System (arXiv)](https://arxiv.org/abs/1603.02754)
- [Wikipedia: Gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
- [The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)](https://hastie.su.domains/ElemStatLearn/)

---

<a id="q36"></a>
### 36. Wyjaśnij, czym regresja logistyczna różni się od regresji liniowej.

**Odpowiedź:**

Obie należą do rodziny modeli liniowych, ale rozwiązują **różne zadania**:

| Aspekt | Regresja liniowa | Regresja logistyczna |
|---|---|---|
| Zadanie | regresja (wartość ciągła) | klasyfikacja (prawdopodobieństwo klasy) |
| Wyjście | $\hat y = w^\top x + b \in \mathbb{R}$ | $\hat p = \sigma(w^\top x + b) \in (0,1)$ |
| Funkcja straty | MSE (najmniejsze kwadraty) | log-loss (cross-entropy) |
| Założenie o szumie | Gaussowski | Bernoulliego (rozkład etykiety) |
| Estymacja | rozwiązanie zamknięte (równania normalne) | iteracyjna (MLE: gradient descent, L-BFGS, IRLS) |
| Interpretacja wag | zmiana $y$ o $w_j$ na jednostkę $x_j$ | zmiana log-odds o $w_j$; $e^{w_j}$ = odds ratio |

**Dlaczego nie użyć regresji liniowej do klasyfikacji?** Wyjście nie jest ograniczone do [0,1], nie jest interpretowalne jako prawdopodobieństwo, a MSE jest wrażliwe na punkty odstające po "właściwej" stronie granicy, co przesuwa próg decyzyjny. Regresja logistyczna modeluje **log-odds** liniowo:

$$\log\frac{p}{1-p} = w^\top x + b$$

**Wspólne cechy:** liniowa granica decyzyjna (w przestrzeni cech), potrzeba skalowania przy regularyzacji, wrażliwość na współliniowość, możliwość regularyzacji L1/L2. Regresja logistyczna to uogólniony model liniowy (GLM) z funkcją łączącą logit; regresja liniowa - GLM z funkcją tożsamościową.

**Uwaga:** mimo nazwy regresja logistyczna jest klasyfikatorem; próg (domyślnie 0.5) dobieramy do kosztów błędów.

**Źródła:**
- [Outcome School: Linear Regression vs Logistic Regression](https://outcomeschool.com/blog/linear-regression-vs-logistic-regression)
- [scikit-learn: Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
- [Wikipedia: Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Wikipedia: Generalized linear model](https://en.wikipedia.org/wiki/Generalized_linear_model)

---

<a id="q37"></a>
### 37. Jak działa regresja logistyczna?

**Odpowiedź:**

Regresja logistyczna to liniowy klasyfikator probabilistyczny. Dla klasyfikacji binarnej:

1. Oblicz wynik liniowy (logit): $z = w^\top x + b$.
2. Zamień go na prawdopodobieństwo funkcją sigmoidalną: $p(y=1\mid x)=\sigma(z)=\dfrac{1}{1+e^{-z}}$.
3. Zdecyduj: $\hat y = 1$, gdy $p \ge t$ (domyślnie $t=0.5$, co odpowiada $z\ge 0$).

**Trening** - maksymalizacja wiarygodności (MLE), czyli minimalizacja log-loss:

$$L(w,b) = -\frac{1}{N}\sum_{i}\big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$$

Gradient ma elegancką postać: $\nabla_w L = \frac{1}{N}\sum_i (p_i - y_i)x_i$. Funkcja straty jest **wypukła**, więc nie ma lokalnych minimów; optymalizujemy gradient descent, L-BFGS lub Newtonem (IRLS). Przy liniowo separowalnych danych wagi dążą do nieskończoności - dlatego stosuje się **regularyzację** (L2 domyślnie w sklearn, parametr `C = 1/λ`; L1 daje rzadkie wagi).

**Wieloklasowo:** softmax (multinomial) zamiast sigmoidy lub schemat one-vs-rest.

**Interpretacja:** $e^{w_j}$ to mnożnik odds przy wzroście cechy $x_j$ o 1 (przy pozostałych stałych).

**Założenia i pitfalle:** liniowa zależność log-odds od cech (nieliniowość: cechy wielomianowe, interakcje), brak silnej współliniowości, skalowanie cech przy regularyzacji, niezbalansowane klasy (`class_weight`, dobór progu), kalibracja prawdopodobieństw zwykle dobra, ale warto sprawdzić.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
```

**Źródła:**
- [scikit-learn: Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Wikipedia: Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression)
- [Stanford CS229 lecture notes](https://cs229.stanford.edu/)

---

<a id="q38"></a>
### 38. Wyjaśnij R-kwadrat i skorygowany R-kwadrat.

**Odpowiedź:**

**R² (współczynnik determinacji)** mierzy, jaką część wariancji zmiennej objaśnianej wyjaśnia model:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1-\frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}$$

- $R^2=1$ - model idealny, $R^2=0$ - model nie lepszy od stałej predykcji średniej $\bar y$.
- Na zbiorze testowym może być **ujemny** (model gorszy od średniej).
- Dla regresji liniowej z wyrazem wolnym na zbiorze treningowym R² rośnie **zawsze** (nie maleje) po dodaniu dowolnej cechy - nawet losowego szumu.

**Adjusted R²** karze za liczbę predyktorów $p$ przy $n$ obserwacjach:

$$R^2_{adj} = 1 - (1-R^2)\frac{n-1}{n-p-1}$$

Rośnie tylko wtedy, gdy nowa cecha poprawia model bardziej, niż wynikałoby to z przypadku; może spaść i być ujemny.

**Przykład:** $n=50$, $p=5$, $R^2=0.80$: $R^2_{adj}=1-0.2\cdot\frac{49}{44}\approx 0.777$.

**Kiedy używać:** adjusted R² do porównywania modeli liniowych o różnej liczbie cech; do porównań predykcyjnych lepiej walidacja krzyżowa (RMSE/MAE na out-of-sample).

**Ograniczenia:** wysoki R² nie oznacza dobrego modelu (overfitting, zależności nieliniowe, brak spełnienia założeń); nie mówi o przyczynowości; wrażliwy na outliery; nie porównuje modeli z różnymi transformacjami $y$ (np. log). Warto zawsze patrzeć na wykres reszt.

**Źródła:**
- [Wikipedia: Coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- [scikit-learn: r2_score / Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score)
- [statsmodels: OLS regression results](https://www.statsmodels.org/stable/regression.html)

---

<a id="q39"></a>
### 39. Jak sprawdzić współliniowość (multicollinearity) w modelach regresji?

**Odpowiedź:**

**Multicollinearity** występuje, gdy cechy są silnie skorelowane liniowo. Skutki: niestabilne, "skaczące" współczynniki, duże błędy standardowe, trudna interpretacja (predykcje mogą pozostać dobre). Dla macierzy $X^\top X$ bliskiej osobliwej odwrócenie jest numerycznie niestabilne.

**Metody wykrywania:**
1. **Macierz korelacji / heatmapa** - szybka, ale wykrywa tylko zależności parami (|r| > 0.8-0.9 to sygnał).
2. **VIF (Variance Inflation Factor)** - dla cechy $j$ regresujemy ją na pozostałych cechach:
$$VIF_j = \frac{1}{1-R_j^2}$$
Reguła kciuka: VIF > 5 podejrzane, > 10 poważny problem.
3. **Tolerancja** $=1/VIF$.
4. **Condition number** macierzy $X$ / wartości własne $X^\top X$ - duży (> 30) wskazuje współliniowość; bliskie zeru wartości własne.
5. Niestabilność współczynników przy resamplingu/dodaniu cechy, nieistotne p-value mimo istotnego modelu.

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
Xc = add_constant(X)
vif = pd.Series([variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])],
                index=X.columns)
```

**Jak naprawiać:** usunięcie/połączenie skorelowanych cech, PCA, regularyzacja (Ridge stabilizuje, Lasso wybiera), centrowanie cech przy termach wielomianowych, zebranie więcej danych. Uwaga: pułapka zmiennych dummy (dummy variable trap) - pomiń jedną kategorię.

**Źródła:**
- [Wikipedia: Multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)
- [Wikipedia: Variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor)
- [statsmodels: variance_inflation_factor](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)

---

<a id="q40"></a>
### 40. Jak działa algorytm K-najbliższych sąsiadów (KNN)?

**Odpowiedź:**

**KNN** to metoda nieparametryczna, oparta na instancjach ("leniwa" - brak fazy treningu poza zapamiętaniem danych). Dla nowego punktu $x$:

1. Oblicz odległość do wszystkich punktów treningowych (np. euklidesowa $\sqrt{\sum (x_j-x'_j)^2}$, Manhattan, cosinusowa, Minkowski).
2. Wybierz $k$ najbliższych.
3. **Klasyfikacja:** głosowanie większościowe (opcjonalnie ważone $1/d$); **regresja:** średnia (lub ważona średnia) wartości sąsiadów.

**Dobór $k$:** małe $k$ - niski bias, wysoka wariancja (overfitting, wrażliwość na szum); duże $k$ - gładka granica, ryzyko underfittingu. Dobieramy walidacją krzyżową, zwykle nieparzyste $k$ dla klasyfikacji binarnej.

**Kluczowe uwagi:**
- **Skalowanie cech jest obowiązkowe** (standaryzacja/min-max), inaczej dominuje cecha o dużej skali.
- **Klątwa wymiarowości:** w wysokich wymiarach odległości się "wyrównują", KNN działa gorzej - redukcja wymiarów, selekcja cech.
- **Koszt predykcji:** $O(N d)$ naiwnie; przyspieszenie: KD-tree, Ball tree (niskie wymiary), przybliżone wyszukiwanie (HNSW, FAISS, Annoy).
- Pamięć: przechowuje cały zbiór.
- Cechy kategoryczne: odległość Hamminga/Gowera.
- Niezbalansowane klasy: głosowanie ważone.

**Zalety:** prosty, brak założeń o rozkładzie, naturalnie wieloklasowy, elastyczne granice. **Wady:** wolna predykcja, wrażliwość na skalę i nieistotne cechy.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))
```

**Źródła:**
- [scikit-learn: Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [Wikipedia: k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- [Stanford CS231n: Nearest Neighbor Classifier](https://cs231n.github.io/classification/)

---

<a id="q41"></a>
### 41. Wyjaśnij K-Means Clustering. Jak działa? Jakie ma ograniczenia?

**Odpowiedź:**

**K-Means** to algorytm klasteryzacji (uczenie nienadzorowane), dzielący dane na $K$ klastrów przez minimalizację sumy kwadratów odległości punktów od centroidów (inertia, WCSS):

$$J=\sum_{k=1}^{K}\sum_{x_i\in C_k}\lVert x_i-\mu_k\rVert^2$$

**Algorytm (Lloyd):**
1. Zainicjuj $K$ centroidów (najlepiej **k-means++**, który rozrzuca je proporcjonalnie do kwadratu odległości).
2. **Przypisanie:** każdy punkt do najbliższego centroidu.
3. **Aktualizacja:** centroid = średnia punktów klastra.
4. Powtarzaj do zbieżności (brak zmian przypisań lub mała zmiana $J$).

Każdy krok nie zwiększa $J$, więc algorytm zbiega, ale tylko do **minimum lokalnego** - stąd wiele restartów (`n_init`). Złożoność ok. $O(N K d\, T)$.

**Dobór $K$:** metoda łokcia (elbow), silhouette score, gap statistic, wiedza dziedzinowa; przy rozmytych przypadkach BIC w GMM.

**Ograniczenia:**
- Trzeba znać/zgadnąć $K$.
- Zakłada klastry **sferyczne, podobnej wielkości i gęstości** (nie radzi sobie z wydłużonymi, zagnieżdżonymi, o różnej wariancji).
- Wrażliwy na skalę cech (standaryzuj) i **outliery** (średnia).
- Wrażliwy na inicjalizację.
- Słaby w wysokich wymiarach (odległości euklidesowe) - użyj PCA/embeddingów.
- Twarde przypisanie (alternatywa: GMM).
- Tylko cechy numeryczne (dla kategorii: K-Modes, K-Prototypes).

**Alternatywy:** DBSCAN/HDBSCAN (dowolne kształty, szum), GMM, klasteryzacja hierarchiczna, K-Medoids (odporność), Mini-Batch K-Means (duże dane).

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0).fit(X_scaled)
print(km.inertia_, silhouette_score(X_scaled, km.labels_))
```

**Źródła:**
- [scikit-learn: Clustering (K-Means)](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Wikipedia: k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)
- [Wikipedia: k-means++](https://en.wikipedia.org/wiki/K-means%2B%2B)

---

<a id="q42"></a>
### 42. Wyjaśnij Support Vector Machines (SVM). Czym jest kernel trick?

**Odpowiedź:**

**SVM** znajduje hiperpłaszczyznę $w^\top x+b=0$ rozdzielającą klasy z **maksymalnym marginesem** (odległość do najbliższych punktów - **wektorów nośnych**, support vectors). Margines wynosi $2/\lVert w\rVert$, więc maksymalizacja marginesu = minimalizacja $\lVert w\rVert^2$.

**Soft margin** (dane nieseparowalne): 

$$\min_{w,b,\xi}\ \tfrac12\lVert w\rVert^2 + C\sum_i \xi_i \quad \text{s.t.}\ y_i(w^\top x_i+b)\ge 1-\xi_i,\ \xi_i\ge 0$$

Równoważnie strata **hinge**: $\max(0,1-y_i f(x_i))$ + regularyzacja. $C$ steruje kompromisem: duże $C$ - mały margines, mało błędów (ryzyko overfittingu); małe $C$ - szerszy margines, więcej tolerancji.

**Kernel trick:** w postaci dualnej dane występują tylko jako iloczyny skalarne $x_i^\top x_j$. Zastępujemy je funkcją jądra $K(x_i,x_j)=\phi(x_i)^\top\phi(x_j)$, co daje liniową separację w (potencjalnie nieskończenie wymiarowej) przestrzeni cech $\phi(x)$ **bez jawnego obliczania** $\phi$. Predykcja: $f(x)=\sum_i \alpha_i y_i K(x_i,x)+b$ (tylko wektory nośne mają $\alpha_i>0$).

Popularne jądra:
- liniowe: $x^\top x'$ (dane tekstowe, wysokie wymiary),
- wielomianowe: $(\gamma x^\top x'+r)^d$,
- **RBF (Gaussa):** $\exp(-\gamma\lVert x-x'\rVert^2)$ - domyślny wybór; duże $\gamma$ - wąskie, złożone granice (overfitting).

**Praktyka:** skalowanie cech obowiązkowe; strojenie $C$ i $\gamma$ (grid/random search w skali logarytmicznej); trening ok. $O(N^2)$-$O(N^3)$, więc słabo skaluje się do bardzo dużych zbiorów (wtedy `LinearSVC`, SGD, aproksymacje jądra: Nystroem). Brak natywnych prawdopodobieństw (Platt scaling). SVR - wariant regresyjny.

**Źródła:**
- [scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
- [Wikipedia: Support vector machine](https://en.wikipedia.org/wiki/Support_vector_machine)
- [Wikipedia: Kernel method](https://en.wikipedia.org/wiki/Kernel_method)
- [Stanford CS229: SVM notes](https://cs229.stanford.edu/)

---

<a id="q43"></a>
### 43. Czym jest granica decyzyjna (decision boundary) w klasyfikatorach?

**Odpowiedź:**

**Granica decyzyjna** to zbiór punktów w przestrzeni cech, w których klasyfikator jest "niezdecydowany" - przypisanie zmienia się z jednej klasy na drugą. Dla klasyfikatora binarnego z wynikiem $f(x)$ jest to $\{x: f(x)=0\}$, a dla probabilistycznego $\{x: p(y=1\mid x)=t\}$ (zwykle $t=0.5$).

**Kształt zależy od modelu:**
- **Regresja logistyczna, liniowy SVM, LDA, Perceptron** - hiperpłaszczyzna (liniowa).
- **SVM z jądrem RBF, sieci neuronowe** - nieliniowa, gładka.
- **Drzewa decyzyjne** - schodkowa, prostopadła do osi (axis-aligned) i podzielona na prostokąty; lasy - wygładzona wersja.
- **KNN** - nieregularna, wieloboczna (diagram Voronoia dla $k=1$).
- **Naive Bayes / QDA** - kwadratowa (przy Gaussowskich klasach).

**Znaczenie praktyczne:**
- Złożoność granicy odpowiada bias-variance: zbyt prosta - underfitting, zbyt poszarpana - overfitting.
- Zmiana progu $t$ przesuwa granicę (kompromis precision/recall, krzywa ROC).
- Margines (odległość od granicy) niesie informację o pewności modelu.
- Wizualizacja (2D) pomaga diagnozować model, np. `DecisionBoundaryDisplay`.
- Dla wielu klas granice wyznaczają regiony, w których dana klasa ma największy wynik (argmax).

```python
from sklearn.inspection import DecisionBoundaryDisplay
DecisionBoundaryDisplay.from_estimator(clf, X[:, :2], response_method="predict", alpha=0.4)
```

**Źródła:**
- [Wikipedia: Decision boundary](https://en.wikipedia.org/wiki/Decision_boundary)
- [scikit-learn: DecisionBoundaryDisplay](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html)
- [scikit-learn: Classifier comparison example](https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)

---

<a id="q44"></a>
### 44. Wyjaśnij Naive Bayes.

**Odpowiedź:**

**Naive Bayes** to generatywny klasyfikator probabilistyczny oparty na twierdzeniu Bayesa z "naiwnym" założeniem **warunkowej niezależności cech przy danej klasie**:

$$P(y\mid x_1,\dots,x_n)\propto P(y)\prod_{j=1}^{n}P(x_j\mid y)$$

Predykcja: $\hat y=\arg\max_y\ \big[\log P(y)+\sum_j\log P(x_j\mid y)\big]$ (logarytmy zapobiegają underflow).

**Warianty** (różne modele $P(x_j\mid y)$):
- **GaussianNB** - cechy ciągłe, rozkład normalny per klasa,
- **MultinomialNB** - liczności słów (bag-of-words, TF-IDF),
- **BernoulliNB** - cechy binarne,
- **CategoricalNB** - cechy kategoryczne.

**Wygładzanie Laplace'a** (`alpha`) zapobiega zerowym prawdopodobieństwom dla nieobserwowanych kombinacji: $P(x_j\mid y)=\frac{count+\alpha}{N_y+\alpha V}$.

**Zalety:** bardzo szybki trening i predykcja (jedno przejście po danych), działa przy małej liczbie próbek i w wysokich wymiarach, dobry baseline (spam, klasyfikacja tekstu), łatwy w aktualizacji online.

**Wady:** założenie niezależności rzadko prawdziwe (skorelowane cechy są "liczone podwójnie"), słabo skalibrowane prawdopodobieństwa (skrajne wartości; ranking często dobry), nie uczy interakcji cech.

**Uwagi:** jako klasyfikator często działa zaskakująco dobrze, bo do poprawnej decyzji potrzebny jest tylko właściwy ranking klas, a nie dokładne prawdopodobieństwa. Kalibracja: `CalibratedClassifierCV`.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
clf = make_pipeline(TfidfVectorizer(), MultinomialNB(alpha=0.5)).fit(texts, labels)
```

**Źródła:**
- [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Wikipedia: Naive Bayes classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
- [Stanford IR Book: Naive Bayes text classification](https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html)

---

<a id="q45"></a>
### 45. Czym jest redukcja wymiarowości (Dimensionality Reduction)?

**Odpowiedź:**

**Redukcja wymiarowości** to przekształcenie danych z $d$ wymiarów do $k \ll d$ przy zachowaniu jak największej ilości istotnej informacji.

**Po co:**
- **Klątwa wymiarowości** - w wysokich wymiarach dane są rzadkie, odległości tracą sens, rośnie ryzyko overfittingu,
- szybszy trening i mniejsze zużycie pamięci,
- usuwanie szumu i redundancji (współliniowość),
- **wizualizacja** (2D/3D),
- kompresja.

**Dwa podejścia:**
1. **Selekcja cech** (feature selection) - wybór podzbioru oryginalnych cech.
2. **Ekstrakcja cech** (feature extraction) - tworzenie nowych cech:
   - **Liniowe:** PCA (maksymalizacja wariancji), LDA (maksymalizacja separacji klas, nadzorowana), SVD/LSA, Random Projection.
   - **Nieliniowe (manifold learning):** t-SNE, UMAP (wizualizacja; nie do wejścia modeli bez ostrożności), Isomap, LLE, kernel PCA.
   - **Neuronowe:** autoenkodery, VAE, embeddingi.

**Uwagi praktyczne:**
- Fituj redukcję **tylko na zbiorze treningowym** (uniknięcie data leakage) - najlepiej w pipeline.
- Skaluj dane przed PCA.
- t-SNE/UMAP: odległości między klastrami i rozmiary klastrów nie są interpretowalne; wyniki zależą od hiperparametrów (perplexity, n_neighbors).
- Redukcja traci interpretowalność cech i może usunąć informację przydatną dla zadania (szczególnie PCA - nienadzorowana).
- Wybór $k$: skumulowana wyjaśniona wariancja (np. 95%), błąd rekonstrukcji, jakość zadania końcowego w CV.

**Źródła:**
- [scikit-learn: Decomposing signals in components (PCA, SVD, NMF)](https://scikit-learn.org/stable/modules/decomposition.html)
- [scikit-learn: Manifold learning](https://scikit-learn.org/stable/modules/manifold.html)
- [Wikipedia: Dimensionality reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction)
- [Wikipedia: Curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality)

---

<a id="q46"></a>
### 46. Wyjaśnij PCA (Principal Component Analysis). Jak działa? Kiedy go użyć?

**Odpowiedź:**

**PCA** znajduje ortogonalne kierunki (składowe główne), wzdłuż których dane mają największą wariancję, i rzutuje na nie dane. To liniowa, nienadzorowana redukcja wymiarowości.

**Algorytm:**
1. **Wycentruj** dane (odejmij średnią), zwykle też wystandaryzuj.
2. Oblicz macierz kowariancji $\Sigma=\frac{1}{N-1}X^\top X$.
3. Wyznacz wartości i wektory własne: $\Sigma v_i=\lambda_i v_i$ (w praktyce SVD: $X=U S V^\top$, $\lambda_i=s_i^2/(N-1)$).
4. Posortuj malejąco wg $\lambda_i$, wybierz $k$ pierwszych wektorów $V_k$.
5. Rzutuj: $Z=XV_k$.

Wyjaśniona wariancja składowej $i$: $\lambda_i/\sum_j\lambda_j$. PCA minimalizuje też błąd rekonstrukcji (najlepsza rzędu-$k$ aproksymacja w sensie najmniejszych kwadratów).

**Wybór $k$:** próg skumulowanej wariancji (np. 90-95%), wykres osypiska (scree plot), CV zadania końcowego.

**Kiedy używać:** wstępna kompresja i odszumianie, usuwanie współliniowości, wizualizacja, przyspieszenie modeli (KNN, SVM), dekorelacja cech.

**Kiedy nie / ograniczenia:**
- zakłada **liniowe** zależności (inaczej Kernel PCA, UMAP, autoenkoder),
- wrażliwe na **skalę** i outliery,
- wariancja $\ne$ informacja predykcyjna - ważny sygnał może leżeć w kierunku o małej wariancji,
- składowe trudne do interpretacji,
- fit tylko na train.

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
pipe = make_pipeline(StandardScaler(), PCA(n_components=0.95))
Z = pipe.fit_transform(X_train)
print(pipe[-1].explained_variance_ratio_)
```

**Źródła:**
- [scikit-learn: PCA](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [Wikipedia: Principal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)
- [A Tutorial on Principal Component Analysis (Shlens, arXiv)](https://arxiv.org/abs/1404.1100)

---

<a id="q47"></a>
### 47. Wyjaśnij Gradient Descent i jego warianty.

**Odpowiedź:**

**Gradient descent** minimalizuje funkcję straty $L(\theta)$, iteracyjnie przesuwając parametry w kierunku ujemnego gradientu:

$$\theta_{t+1}=\theta_t-\eta\,\nabla_\theta L(\theta_t)$$

gdzie $\eta$ to learning rate. Gradient wskazuje kierunek największego wzrostu, więc krok w przeciwnym kierunku zmniejsza stratę (dla dostatecznie małego $\eta$).

**Warianty ze względu na ilość danych w kroku:**
- **Batch GD** - gradient z całego zbioru: stabilny, wolny, wymaga całych danych w pamięci.
- **Stochastic GD (SGD)** - jedna próbka: szybkie kroki, duży szum (może pomóc uciec z płytkich minimów).
- **Mini-batch GD** - kompromis (32-4096 próbek): standard w deep learningu, efektywny na GPU.

**Warianty z adaptacją kroku:**
- **Momentum:** $v_t=\beta v_{t-1}+\nabla L,\ \theta\leftarrow\theta-\eta v_t$ - tłumi oscylacje, przyspiesza w spójnych kierunkach; **Nesterov** - gradient w punkcie "z wyprzedzeniem".
- **AdaGrad:** learning rate skalowany odwrotnie do sumy kwadratów gradientów (dobre dla rzadkich cech; lr maleje do zera).
- **RMSProp:** wykładnicza średnia kwadratów gradientów.
- **Adam:** momentum + RMSProp z korekcją biasu: $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$, $v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$, $\theta\leftarrow\theta-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. **AdamW** - poprawnie oddzielony weight decay (standard dla Transformerów).

**Praktyka:** lr schedules (warmup, cosine, step), gradient clipping, skalowanie cech, dobra inicjalizacja. SGD+momentum często daje lepszą generalizację w wizji; Adam/AdamW zbiega szybciej i jest domyślny w NLP.

```python
import torch
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step(); opt.zero_grad()
```

**Źródła:**
- [Outcome School: Math Behind Gradient Descent](https://outcomeschool.com/blog/math-behind-gradient-descent)
- [An overview of gradient descent optimization algorithms (Ruder, arXiv)](https://arxiv.org/abs/1609.04747)
- [Adam: A Method for Stochastic Optimization (arXiv)](https://arxiv.org/abs/1412.6980)
- [Deep Learning book, ch. 8: Optimization](https://www.deeplearningbook.org/contents/optimization.html)

---

<a id="q48"></a>
### 48. Czym jest krzywa ROC-AUC i jak ją interpretować?

**Odpowiedź:**

**Krzywa ROC** (Receiver Operating Characteristic) pokazuje kompromis klasyfikatora binarnego dla wszystkich możliwych progów decyzyjnych:
- oś Y: **TPR** (recall, czułość) $=\frac{TP}{TP+FN}$,
- oś X: **FPR** $=\frac{FP}{FP+TN}=1-\text{swoistość}$.

Obniżanie progu zwiększa zarówno TPR, jak i FPR - krzywa idzie z (0,0) do (1,1).

**AUC** (Area Under the Curve) to pole pod krzywą, wartość 0-1:
- 1.0 - idealne rozdzielenie,
- 0.5 - losowy klasyfikator (przekątna),
- < 0.5 - odwrócone przewidywania (odwróć znak).

**Interpretacja probabilistyczna:** AUC to prawdopodobieństwo, że losowy przykład pozytywny dostanie wyższy wynik niż losowy negatywny (równoważne statystyce Manna-Whitneya/Wilcoxona).

**Zalety:** niezależna od progu, niezależna od skali/kalibracji wyników (liczy się tylko ranking), niewrażliwa na proporcje klas (w pewnym sensie).

**Wady/pułapki:**
- Przy **silnym niezbalansowaniu** ROC-AUC może być zbyt optymistyczny (duża liczba TN utrzymuje FPR niskim) - lepsza **krzywa Precision-Recall / PR-AUC (Average Precision)**.
- Nie mówi nic o kalibracji ani o zachowaniu w konkretnym progu/regionie istotnym biznesowo (można użyć partial AUC).
- Nie uwzględnia kosztów błędów FP i FN.
- Punkt pracy wybieramy osobno (np. Youden's J $=TPR-FPR$, punkt spełniający wymagania biznesowe).

Wieloklasowo: one-vs-rest lub one-vs-one, uśrednianie macro/weighted.

```python
from sklearn.metrics import roc_auc_score, roc_curve
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
fpr, tpr, thr = roc_curve(y_test, scores)
```

**Źródła:**
- [scikit-learn: ROC metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
- [Wikipedia: Receiver operating characteristic](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [An introduction to ROC analysis (Fawcett, 2006)](https://doi.org/10.1016/j.patrec.2005.10.010)
- [The Relationship Between Precision-Recall and ROC Curves (Davis & Goadrich)](https://dl.acm.org/doi/10.1145/1143844.1143874)

---

## Przygotowanie danych i inżynieria cech

<a id="q49"></a>
### 49. Czym jest Feature Engineering?

**Odpowiedź:**

**Feature engineering** to proces tworzenia, przekształcania i wybierania cech (zmiennych wejściowych) tak, aby algorytm uczenia maszynowego mógł jak najlepiej wykorzystać informację zawartą w danych. Dla danych tabelarycznych często ma większy wpływ na jakość niż wybór algorytmu.

**Główne działania:**
- **Czyszczenie:** braki, outliery, duplikaty, błędy.
- **Transformacje:** log/Box-Cox/Yeo-Johnson dla skośnych rozkładów, skalowanie, dyskretyzacja (binning).
- **Kodowanie kategorii:** one-hot, ordinal, target encoding, hashing.
- **Tworzenie cech:** interakcje i stosunki (`cena/m2`), agregacje grupowe (średnia zakupów na klienta), cechy wielomianowe, cechy czasowe (dzień tygodnia, sin/cos dla cykliczności, lagi, średnie kroczące), cechy tekstowe (TF-IDF, długość, embeddingi), geograficzne (odległości).
- **Selekcja i redukcja:** filtrowanie, metody wrapper/embedded, PCA.
- **Wiedza dziedzinowa** - najcenniejsze cechy zwykle wynikają ze zrozumienia problemu.

**Pułapki:**
- **Data leakage** - cechy używające informacji z przyszłości/etykiety lub statystyki liczone na całym zbiorze przed podziałem. Wszystkie transformacje fituj na train, w `Pipeline`.
- Spójność train/serving (training-serving skew) - ten sam kod w produkcji, ewentualnie feature store.
- Nadmiar cech - overfitting, koszt utrzymania.

Deep learning częściowo automatyzuje ten proces (representation learning), ale w danych tabelarycznych ręczne cechy nadal się opłacają.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)])
```

**Źródła:**
- [Feature Engineering in Machine Learning (film, źródło z README)](https://www.youtube.com/watch?v=QLlywrWuXag)
- [Feature Engineering and Selection (Kuhn & Johnson)](https://www.feat.engineering/)
- [scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Wikipedia: Feature engineering](https://en.wikipedia.org/wiki/Feature_engineering)

---

<a id="q50"></a>
### 50. Czym jest kodowanie one-hot (one-hot encoding)? Kiedy go stosować?

**Odpowiedź:**

**One-hot encoding** zamienia zmienną kategoryczną o $K$ wartościach na $K$ binarnych kolumn; dokładnie jedna z nich ma wartość 1 (dla danej kategorii), pozostałe 0. Np. `kolor ∈ {czerwony, zielony, niebieski}` -> `[1,0,0]`, `[0,1,0]`, `[0,0,1]`.

**Dlaczego:** większość modeli (regresja, sieci, SVM, KNN) wymaga liczb, a proste etykietowanie liczbami (1, 2, 3) narzuca **fałszywy porządek i odległości** (np. że niebieski = 3 × czerwony).

**Kiedy stosować:**
- kategorie **nominalne** (bez naturalnego porządku),
- mała lub średnia liczba unikalnych wartości (kardynalność),
- modele liniowe, sieci neuronowe, KNN, SVM.

**Kiedy uważać / unikać:**
- **Wysoka kardynalność** (ID, kod pocztowy, tysiące wartości) - eksplozja wymiarów i rzadkie macierze; alternatywy: target encoding, hashing trick, embeddingi, grupowanie rzadkich kategorii w "other".
- Kategorie **porządkowe** (ordinal) - lepsze kodowanie porządkowe.
- Modele drzewiaste - działają, ale przy wielu kategoriach mogą tracić wydajność; LightGBM/CatBoost obsługują kategorie natywnie.

**Pułapki:**
- **Dummy variable trap:** kolumny są liniowo zależne (suma = 1) - w regresji liniowej usuń jedną (`drop="first"`).
- Nieznane kategorie w produkcji: `handle_unknown="ignore"`.
- Fit encodera tylko na train, w pipeline.

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_cat = enc.fit_transform(df_train[["kolor"]])
```

**Źródła:**
- [One-hot Encoding in Machine Learning (film, źródło z README)](https://www.youtube.com/watch?v=6AmedU5i9go)
- [scikit-learn: OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
- [Wikipedia: One-hot](https://en.wikipedia.org/wiki/One-hot)

---

<a id="q51"></a>
### 51. Jak radzić sobie z brakującymi danymi?

**Odpowiedź:**

Najpierw zrozum **mechanizm braków** (Rubin):
- **MCAR** - całkowicie losowe (brak związku z danymi),
- **MAR** - zależą od innych zaobserwowanych zmiennych,
- **MNAR** - zależą od samej brakującej wartości (np. wysokie dochody nie są podawane). Najtrudniejszy; samo imputowanie może wprowadzać bias.

Sprawdź też skalę braków i to, czy sama informacja "brak" niesie sygnał.

**Strategie:**
1. **Usuwanie** wierszy (listwise) - tylko gdy braków mało i MCAR; usuwanie kolumn przy np. > 50-60% braków (jeśli nieistotne).
2. **Imputacja prostą statystyką:** średnia/mediana (odporna na outliery), moda dla kategorii, stała ("Unknown"). Szybka, ale zaniża wariancję i osłabia korelacje.
3. **Imputacja zależna od kontekstu:** średnia w grupie, forward/backward fill i interpolacja dla szeregów czasowych.
4. **KNN imputer**, **IterativeImputer** (MICE - regresja iteracyjna), **multiple imputation** (uwzględnia niepewność).
5. **Modele obsługujące braki natywnie:** XGBoost, LightGBM, HistGradientBoosting.
6. **Wskaźnik braku** (`add_indicator=True`): dodatkowa kolumna binarna - często bardzo pomocna, szczególnie przy MNAR.

**Zasady:** imputer fituj wyłącznie na zbiorze treningowym (uniknięcie leakage), używaj w pipeline; zapewnij tę samą logikę w produkcji; oceń wpływ metody imputacji w walidacji krzyżowej.

```python
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
num = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler())
```

**Źródła:**
- [scikit-learn: Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)
- [Wikipedia: Missing data](https://en.wikipedia.org/wiki/Missing_data)
- [Wikipedia: Imputation (statistics)](https://en.wikipedia.org/wiki/Imputation_(statistics))

---

<a id="q52"></a>
### 52. Jak radzić sobie z wartościami odstającymi (outliers)?

**Odpowiedź:**

**Outlier** to obserwacja znacznie odbiegająca od reszty. Może być **błędem** (pomiar, wprowadzanie danych), **rzadkim, ale prawdziwym zdarzeniem** (oszustwo, awaria - często cenne) lub skutkiem innej populacji. Kluczowe: najpierw zrozum przyczynę, nie usuwaj automatycznie.

**Wykrywanie:**
- **Wizualne:** box plot, histogram, scatter plot.
- **Reguła IQR:** poza $[Q_1-1.5\,IQR,\ Q_3+1.5\,IQR]$.
- **Z-score:** $|z|>3$ (zakłada zbliżoną normalność; wrażliwy na same outliery - lepszy **modified z-score** z medianą i MAD).
- **Wielowymiarowe:** odległość Mahalanobisa, Isolation Forest, Local Outlier Factor, One-Class SVM, DBSCAN.
- Miary wpływu w regresji: odległość Cooka, dźwignia (leverage).

**Postępowanie:**
1. **Poprawa/usunięcie** - jeśli to jednoznaczny błąd.
2. **Winsoryzacja / clipping** do percentyli (np. 1. i 99.).
3. **Transformacje** redukujące skośność: log, pierwiastek, Box-Cox, Yeo-Johnson; `QuantileTransformer`.
4. **Skalowanie odporne:** `RobustScaler` (mediana, IQR).
5. **Odporne modele/straty:** Huber loss, MAE, regresja kwantylowa, RANSAC, drzewa (mało wrażliwe na wartości skrajne).
6. **Osobny model / flaga** dla przypadków skrajnych.

**Uwagi:** wrażliwość zależy od modelu (regresja liniowa, KNN, K-Means bardzo wrażliwe; drzewa i lasy mniej). Stosuj progi wyliczone na train. Gdy zadaniem jest wykrywanie anomalii, outliery są celem, nie problemem.

```python
q1, q3 = df["x"].quantile([.25, .75]); iqr = q3 - q1
df["x_clip"] = df["x"].clip(q1 - 1.5*iqr, q3 + 1.5*iqr)
```

**Źródła:**
- [scikit-learn: Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [scikit-learn: RobustScaler / Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-data-with-outliers)
- [Wikipedia: Outlier](https://en.wikipedia.org/wiki/Outlier)

---

<a id="q53"></a>
### 53. Wyjaśnij skalowanie cech (Feature Scaling). Dlaczego jest potrzebne?

**Odpowiedź:**

**Skalowanie cech** doprowadza cechy do porównywalnych zakresów/rozkładów.

**Główne metody:**
- **Standaryzacja (z-score):** $x'=\frac{x-\mu}{\sigma}$ - średnia 0, odchylenie 1. Domyślny wybór dla większości modeli.
- **Min-Max:** $x'=\frac{x-x_{min}}{x_{max}-x_{min}}\in[0,1]$ - przy ograniczonych zakresach (piksele, sieci z sigmoidą); wrażliwe na outliery.
- **RobustScaler:** $\frac{x-\text{mediana}}{IQR}$ - odporne na outliery.
- **MaxAbsScaler** - zachowuje rzadkość.
- **Transformacje nieliniowe:** log, PowerTransformer, QuantileTransformer.
- **Normalizacja wektora** (L2 na wierszu) - dla cosinusowych podobieństw.

**Dlaczego potrzebne:**
- Modele **oparte na odległościach** (KNN, K-Means, SVM RBF) - cecha o dużej skali dominuje.
- Modele **gradientowe** (regresja, sieci neuronowe) - lepiej uwarunkowana funkcja straty, szybsza i stabilniejsza zbieżność (spłaszczone "doliny").
- **Regularyzacja** L1/L2 - kara zależy od skali wag, więc bez skalowania jest niesprawiedliwa.
- **PCA** - zależy od wariancji.

**Niepotrzebne** dla modeli drzewiastych (drzewa, RF, GBM) - niewrażliwe na monotoniczne przekształcenia.

**Zasady:** `fit` na train, `transform` na val/test (inaczej leakage); zapisz scaler razem z modelem; nie skaluj zmiennych binarnych/one-hot bez potrzeby; zmienną docelową w regresji można skalować osobno (pamiętaj o odwróceniu).

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X_train)
X_train_s, X_test_s = sc.transform(X_train), sc.transform(X_test)
```

**Źródła:**
- [scikit-learn: Preprocessing data (scaling)](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling)
- [scikit-learn: Importance of Feature Scaling](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html)
- [Wikipedia: Feature scaling](https://en.wikipedia.org/wiki/Feature_scaling)

---

<a id="q54"></a>
### 54. One-Hot, Label, Target i K-Fold Target Encoding - na czym polegają i czym się różnią?

**Odpowiedź:**

Cztery sposoby zamiany zmiennych kategorycznych na liczby:

| Metoda | Idea | Zalety | Wady |
|---|---|---|---|
| **One-Hot** | kolumna binarna na kategorię | brak fałszywego porządku | wysoka kardynalność = wiele kolumn |
| **Label / Ordinal** | kategoria -> liczba całkowita | 1 kolumna, kompaktowa | narzuca porządek (ok. dla drzew i cech porządkowych) |
| **Target (mean) encoding** | kategoria -> średnia targetu w tej kategorii | 1 kolumna, uchwyca siłę zależności, radzi sobie z wysoką kardynalnością | **ryzyko leakage i overfittingu** |
| **K-Fold Target** | target encoding liczony out-of-fold | ogranicza leakage | bardziej złożony, szum |

**Label/Ordinal:** `{niski:0, średni:1, wysoki:2}` sensowne dla cech porządkowych; dla nominalnych stosować głównie z modelami drzewiastymi.

**Target encoding:** dla kategorii $c$: $\text{enc}(c)=\text{mean}(y \mid x=c)$. Problem: dla rzadkich kategorii średnia jest szumowa, a wiersz "widzi" własną etykietę - model uczy się na wycieku i overfituje. **Wygładzanie** (smoothing) łączy średnią kategorii z globalną:

$$\text{enc}(c)=\frac{n_c\,\bar y_c+m\,\bar y}{n_c+m}$$

**K-Fold Target Encoding:** dzielimy train na $K$ foldów; dla wierszy z folda $k$ liczymy średnie **na pozostałych $K-1$ foldach**. Wiersz nigdy nie wpływa na własne kodowanie. Na zbiór testowy stosujemy średnie z całego train. Dodatkowo szum, regularyzacja. Scikit-learn `TargetEncoder` robi to automatycznie (cross fitting w `fit_transform`).

**Wybór:** mała kardynalność nominalna - one-hot; porządkowe - ordinal; wysoka kardynalność - target encoding (K-Fold + smoothing), hashing lub embeddingi; CatBoost ma własne *ordered target statistics*.

```python
from sklearn.preprocessing import TargetEncoder
te = TargetEncoder(smooth="auto", cv=5)
X_tr = te.fit_transform(X_train[["miasto"]], y_train)   # cross-fitting
X_te = te.transform(X_test[["miasto"]])
```

**Źródła:**
- [scikit-learn: TargetEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html)
- [scikit-learn: Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
- [category_encoders: dokumentacja](https://contrib.scikit-learn.org/category_encoders/)
- [CatBoost: Transforming categorical features to numerical](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic)

---

<a id="q55"></a>
### 55. Jak radzić sobie z cechami kategorycznymi?

**Odpowiedź:**

**Krok 1 - rozpoznaj typ i kardynalność:** nominalne (miasto), porządkowe (rozmiar S/M/L), binarne, wysokiej kardynalności (ID użytkownika), cykliczne (miesiąc, godzina).

**Krok 2 - wybierz kodowanie:**
- **Binarne:** 0/1.
- **Porządkowe:** ordinal encoding zgodny z porządkiem.
- **Nominalne, niska kardynalność:** one-hot.
- **Wysoka kardynalność:** target encoding (K-Fold + smoothing), frequency/count encoding, hashing trick (stała liczba kolumn, możliwe kolizje), embeddingi uczone w sieci neuronowej, grupowanie rzadkich kategorii w "other".
- **Cykliczne:** sin/cos, np. $\sin(2\pi h/24),\ \cos(2\pi h/24)$.
- **Modele z natywną obsługą kategorii:** CatBoost, LightGBM (`categorical_feature`), HistGradientBoosting (`categorical_features`).

**Krok 3 - praktyczne kwestie:**
- **Nieznane kategorie** w produkcji (`handle_unknown`), kategoria "missing" jako osobna wartość.
- **Rzadkie kategorie** - łączenie, aby uniknąć overfittingu.
- **Leakage** - encodery zależne od targetu fituj wyłącznie na train (cross-fitting).
- **Dummy trap** w modelach liniowych.
- Spójność kodowania między train i serwowaniem (pipeline, zapisany encoder).
- Standaryzacja nie jest potrzebna dla one-hot; dla drzew nie trzeba one-hot.

**Wybór a model:** liniowe/NN - one-hot lub embeddingi; drzewa - ordinal/native; KNN/SVM - one-hot z odpowiednią miarą odległości (Gower).

```python
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
ct = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", min_frequency=20), ["kolor"]),
    (OrdinalEncoder(categories=[["S","M","L"]]), ["rozmiar"]))
```

**Źródła:**
- [scikit-learn: Encoding categorical features](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
- [scikit-learn: Categorical support in HistGradientBoosting](https://scikit-learn.org/stable/modules/ensemble.html#categorical-support-in-gradient-boosting)
- [CatBoost: Categorical features](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic)
- [Entity Embeddings of Categorical Variables (Guo & Berkhahn, arXiv)](https://arxiv.org/abs/1604.06737)

---

<a id="q56"></a>
### 56. Selekcja cech (feature selection) a ekstrakcja cech (feature extraction) - jaka jest różnica?

**Odpowiedź:**

Obie techniki zmniejszają liczbę cech, ale różnią się sposobem:

| Aspekt | Feature selection | Feature extraction |
|---|---|---|
| Idea | wybór **podzbioru** oryginalnych cech | **tworzenie nowych** cech jako transformacji istniejących |
| Przykłady | filtry, Lasso, RFE, importance z drzew | PCA, LDA, autoenkodery, t-SNE/UMAP, SVD |
| Interpretowalność | zachowana (cechy mają swoje znaczenie) | zwykle utracona |
| Koszt produkcyjny | mniej zbieranych/obliczanych cech | nadal potrzeba wszystkich cech wejściowych |
| Typ | zawsze podzbiór | liniowe lub nieliniowe kombinacje |

**Metody selekcji:**
- **Filter:** niezależne od modelu - korelacja, chi², ANOVA F-test, informacja wzajemna, próg wariancji.
- **Wrapper:** przeszukiwanie z użyciem modelu - forward/backward selection, RFE (drogie).
- **Embedded:** selekcja w trakcie treningu - regularyzacja L1 (Lasso), ważność cech w drzewach.

**Kiedy co:**
- Selection - gdy interpretowalność, koszt zbierania danych lub regulacje są ważne; wiele nieistotnych cech.
- Extraction - gdy cechy silnie skorelowane, dane wysokowymiarowe (obrazy, tekst, geny), potrzebna kompresja/wizualizacja.

**Uwagi:** selekcja/ekstrakcja powinna być w pipeline i fitowana wyłącznie na train (w CV - w każdym foldzie), inaczej **selection bias / leakage**. Cechy w deep learningu uczone są automatycznie (representation learning).

**Źródła:**
- [scikit-learn: Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [scikit-learn: Decomposition (feature extraction)](https://scikit-learn.org/stable/modules/decomposition.html)
- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, JMLR)](https://jmlr.org/papers/v3/guyon03a.html)

---

<a id="q57"></a>
### 57. Jak tworzyć nowe cechy z istniejących?

**Odpowiedź:**

Nowe cechy powinny uwypuklać zależności, których model sam nie wychwyci lub wychwyci trudno. Zacznij od **wiedzy dziedzinowej** i eksploracji danych (EDA), a potem weryfikuj wartość cechy w walidacji krzyżowej.

**Typowe techniki:**
- **Arytmetyka i stosunki:** `cena/powierzchnia`, `dochód/liczba_osób`, różnice (`data_zakupu - data_rejestracji`), BMI.
- **Interakcje i wielomiany:** $x_1x_2$, $x^2$ (`PolynomialFeatures`) - ważne dla modeli liniowych; drzewa łapią interakcje same.
- **Transformacje:** log, pierwiastek dla skośności; binning (kwantyle); Box-Cox.
- **Agregacje grupowe:** średnia/max/liczba transakcji na klienta, odchylenie od średniej grupy (target encoding, frequency encoding).
- **Cechy czasowe:** rok, miesiąc, dzień tygodnia, weekend/święto, cykliczne sin/cos, lagi, średnie kroczące, okna czasowe, "czas od ostatniego zdarzenia".
- **Tekst:** długość, liczba słów, TF-IDF, n-gramy, embeddingi.
- **Obraz/dźwięk:** cechy z pretrenowanych sieci, spektrogramy.
- **Geograficzne:** odległość do centrum, klastry lokalizacji, geohash.
- **Wskaźniki:** flagi braków, flagi progów (`czy_premium`).
- **Cechy z modeli:** wyniki klasteryzacji, PCA, predykcje z modeli (stacking, uwaga na leakage - out-of-fold).

**Pułapki:**
- **Leakage:** agregaty i statystyki liczone tylko z danych dostępnych w momencie predykcji (szereg czasowy - tylko przeszłość), na train w CV.
- Korelowane, redundantne cechy - selekcja po tworzeniu.
- Spójność w produkcji (feature store, wspólny kod).
- Eksplozja liczby cech - overfitting.

**Weryfikacja:** porównanie metryki CV przed/po, ważność cech, ablacje.

```python
df["cena_m2"] = df["cena"] / df["metraz"]
df["godz_sin"] = np.sin(2*np.pi*df["godzina"]/24)
df["godz_cos"] = np.cos(2*np.pi*df["godzina"]/24)
df["klient_avg"] = df.groupby("klient_id")["kwota"].transform("mean")  # uwaga na leakage
```

**Źródła:**
- [Feature Engineering and Selection (Kuhn & Johnson)](https://www.feat.engineering/)
- [scikit-learn: Preprocessing (PolynomialFeatures, discretization)](https://scikit-learn.org/stable/modules/preprocessing.html)
- [scikit-learn: Time-related feature engineering](https://scikit-learn.org/stable/auto_examples/applications/plot_cyclical_feature_engineering.html)

---

<a id="q58"></a>
### 58. Jak podejść do zbioru danych z silnie niezbalansowanymi klasami?

**Odpowiedź:**

Przy np. 1% pozytywów model "zawsze klasa negatywna" ma 99% accuracy, więc kluczowe są **właściwe metryki** i techniki uwzględniające rzadką klasę.

**1. Metryki:** nie accuracy. Używaj precision, recall, F1/F-beta, **PR-AUC (Average Precision)**, MCC, balanced accuracy, ROC-AUC (ostrożnie), a często metryk kosztowych. Stratyfikowany podział (`stratify=y`, `StratifiedKFold`).

**2. Techniki na poziomie danych:**
- **Oversampling** klasy mniejszościowej: losowy, **SMOTE** (syntetyczne punkty przez interpolację sąsiadów), ADASYN.
- **Undersampling** większościowej: losowy, Tomek links, NearMiss (traci informację).
- Zbieranie/generowanie większej liczby przykładów mniejszościowych, augmentacja.
- Resampling **tylko na zbiorze treningowym** (i wewnątrz foldów CV, np. `imblearn.Pipeline`) - nigdy na walidacyjnym/testowym.

**3. Techniki na poziomie algorytmu:**
- **Wagi klas** (`class_weight="balanced"`, `scale_pos_weight` w XGBoost),
- **Focal loss** (sieci), cost-sensitive learning,
- Modele odporne: drzewa/ensemble (Balanced Random Forest, EasyEnsemble).

**4. Próg decyzyjny:** dostrój próg zamiast 0.5 na krzywej PR pod koszty biznesowe; skalibruj prawdopodobieństwa (resampling je zniekształca).

**5. Inne:** wykrywanie anomalii (skrajna nierównowaga), reformulacja problemu.

**Ostrzeżenia:** SMOTE przed splitem = leakage; resampling nie zawsze poprawia wyniki - zacznij od wag klas i progu; raportuj metryki na oryginalnym rozkładzie.

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline([("smote", SMOTE(random_state=0)),
                 ("clf", RandomForestClassifier(class_weight="balanced"))])
```

**Źródła:**
- [imbalanced-learn: dokumentacja](https://imbalanced-learn.org/stable/)
- [SMOTE: Synthetic Minority Over-sampling Technique (arXiv)](https://arxiv.org/abs/1106.1813)
- [Focal Loss for Dense Object Detection (arXiv)](https://arxiv.org/abs/1708.02002)
- [scikit-learn: Tuning the decision threshold](https://scikit-learn.org/stable/modules/classification_threshold.html)

---

<a id="q59"></a>
### 59. Jak wybierasz cechy dla modelu?

**Odpowiedź:**

Wybór cech poprawia generalizację, skraca trening, upraszcza wdrożenie i interpretację. Sensowny proces:

**1. Wiedza dziedzinowa i EDA** - usuń oczywiste ID, cechy niedostępne w produkcji, cechy powodujące leakage (przyszłość, pochodne targetu).

**2. Filtry (szybkie, niezależne od modelu):**
- próg wariancji (stałe/quasi-stałe cechy),
- korelacja z targetem, korelacja parami (usuń redundantne przy |r| > 0.9),
- testy statystyczne: chi², ANOVA F, informacja wzajemna (`SelectKBest`).

**3. Wrappery (dokładniejsze, kosztowne):** forward/backward selection, **RFE / RFECV** (rekurencyjne odrzucanie najsłabszych cech).

**4. Metody embedded:** regularyzacja **L1** (Lasso, L1-logistic), ważność w drzewach/GBM, ElasticNet.

**5. Ważność cech (feature importance):** **permutation importance** (mierzona na walidacji; bardziej wiarygodna niż importance oparte na nieczystości drzew, które faworyzują cechy o wysokiej kardynalności), **SHAP**. Uwaga na skorelowane cechy.

**6. Ocena:** walidacja krzyżowa metryki dla podzbiorów; kompromis wydajność-liczba cech (reguła "one-standard-error").

**Pułapki:**
- Selekcja na całym zbiorze przed CV = **leakage**; umieść selektor w pipeline.
- Stabilność - sprawdź, czy wybór jest podobny w różnych foldach/próbkach.
- Cechy słabe pojedynczo mogą być silne w interakcji (filtry je pomijają).
- Wybór pod kątem interpretowalności/kosztu, nie tylko metryki.

```python
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
sel = RFECV(LogisticRegression(max_iter=1000), cv=5, scoring="roc_auc").fit(X_train, y_train)
print(X.columns[sel.support_])
```

**Źródła:**
- [scikit-learn: Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)
- [scikit-learn: Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [A Unified Approach to Interpreting Model Predictions (SHAP, arXiv)](https://arxiv.org/abs/1705.07874)
- [An Introduction to Variable and Feature Selection (Guyon & Elisseeff, JMLR)](https://jmlr.org/papers/v3/guyon03a.html)

---

<a id="q60"></a>
### 60. Dlaczego i jak dzielimy dane na zbiór treningowy, testowy i walidacyjny?

**Odpowiedź:**

**Cel:** oszacować, jak model zachowa się na **nowych danych**, i uniknąć oceniania na tym, na czym się uczył (overfitting byłby niewidoczny).

**Role zbiorów:**
- **Train** - uczenie parametrów (wag).
- **Validation** - dobór hiperparametrów, architektury, early stopping, wybór modelu. Wielokrotne użycie "zużywa" go (pośredni overfitting do walidacji).
- **Test** - jednorazowa, końcowa, nieobciążona ocena; nie używany do żadnych decyzji.

**Typowe proporcje:** 60/20/20, 70/15/15, 80/10/10; przy milionach próbek wystarczą małe zbiory val/test (np. 1-2%), przy małych danych - **walidacja krzyżowa** (k-fold, zwykle k=5 lub 10) zamiast stałego zbioru walidacyjnego, z osobnym testem lub nested CV.

**Jak dzielić:**
- **Losowo** z ustalonym `random_state` (dane i.i.d.).
- **Stratyfikowanie** (`stratify=y`) - zachowanie proporcji klas.
- **Grupowo** (`GroupKFold`) - gdy wiele wierszy dotyczy tego samego użytkownika/pacjenta/obiektu, inaczej przeciek między zbiorami.
- **Czasowo** (`TimeSeriesSplit`) - trening na przeszłości, test na przyszłości; nigdy losowo dla szeregów czasowych.
- Deduplikacja przed podziałem; test powinien odzwierciedlać rozkład produkcyjny.

**Pułapki:**
- **Data leakage** - skalowanie, imputacja, selekcja cech, SMOTE, target encoding fitowane przed podziałem; używaj `Pipeline` i fituj tylko na train.
- Dostrajanie na teście, zbyt mały test (duża wariancja oceny), niereprezentatywny podział (data drift).
- Po wyborze hiperparametrów często przetrenowuje się finalny model na train+val.

```python
from sklearn.model_selection import train_test_split
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)
```

**Źródła:**
- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn: Common pitfalls (data leakage)](https://scikit-learn.org/stable/common_pitfalls.html)
- [Stanford CS229: Evaluation and validation notes](https://cs229.stanford.edu/)

---

## Optymalizacja

<a id="q61"></a>
### 61. Czym jest gradient descent? Jak działa?

**Odpowiedź:**

**Gradient descent (spadek gradientowy)** to iteracyjny algorytm optymalizacji pierwszego rzędu, minimalizujący funkcję straty $L(\theta)$. Intuicja: stoisz na wzgórzu we mgle i robisz krok w stronę największego lokalnego spadku.

**Mechanizm:**
1. Zainicjuj parametry $\theta_0$.
2. Oblicz gradient $\nabla_\theta L(\theta_t)$ (w sieciach neuronowych - backpropagation).
3. Zaktualizuj: $\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)$.
4. Powtarzaj do zbieżności (mały gradient, brak poprawy loss).

**Przykład: regresja liniowa z MSE**, $L=\frac1N\sum(y_i-wx_i-b)^2$:
$\frac{\partial L}{\partial w}=-\frac2N\sum x_i(y_i-\hat y_i)$, $\frac{\partial L}{\partial b}=-\frac2N\sum(y_i-\hat y_i)$; następnie $w\leftarrow w-\eta\,\partial L/\partial w$.

**Własności:**
- Dla funkcji wypukłych i odpowiedniego $\eta$ zbiega do globalnego minimum; dla sieci neuronowych (niewypukłe) - do minimum lokalnego/punktu siodłowego, w praktyce zwykle wystarczająco dobrego.
- Za duży $\eta$ - rozbieżność/oscylacje, za mały - powolna zbieżność.
- Zła skala cech (wydłużone "doliny") spowalnia; pomaga standaryzacja, momentum, adaptacyjne optymalizatory.

**Warianty:** batch, mini-batch, stochastic; momentum, RMSProp, Adam. Zaawansowane: metody drugiego rzędu (Newton, L-BFGS).

```python
import numpy as np
w, b, lr = 0.0, 0.0, 0.05
for _ in range(1000):
    err = (w*x + b) - y
    w -= lr * 2*np.mean(err*x)
    b -= lr * 2*np.mean(err)
```

**Źródła:**
- [Outcome School: Math Behind Gradient Descent](https://outcomeschool.com/blog/math-behind-gradient-descent)
- [Stanford CS231n: Optimization](https://cs231n.github.io/optimization-1/)
- [Wikipedia: Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)
- [Deep Learning book, ch. 4.3: Gradient-Based Optimization](https://www.deeplearningbook.org/contents/numerical.html)

---

<a id="q62"></a>
### 62. Czym jest stochastic gradient descent (SGD)?

**Odpowiedź:**

**SGD** aproksymuje gradient pełnej straty gradientem policzonym na **jednej próbce** (lub małym mini-batchu), zamiast na całym zbiorze:

$$\theta_{t+1}=\theta_t-\eta\,\nabla_\theta L(\theta_t;x_i,y_i)$$

Gradient z mini-batcha jest **nieobciążonym estymatorem** pełnego gradientu, ale zaszumionym.

**Zalety:**
- Koszt kroku niezależny od rozmiaru zbioru $N$ - wiele aktualizacji na epokę, szybszy postęp na dużych danych.
- Mieści się w pamięci (dane strumieniowo).
- Szum działa jak regularyzator i pomaga uciekać z płytkich minimów/punktów siodłowych; wiąże się z lepszą generalizacją (płaskie minima).
- Możliwość uczenia online.

**Wady:** oscylacje, brak dokładnej zbieżności przy stałym $\eta$ (kręci się wokół minimum) - stąd **harmonogramy learning rate** (decay, cosine) lub rosnący batch; wrażliwość na dobór $\eta$ i skalę cech.

**Mini-batch** (32-512, w dużych modelach więcej) to standard: zmniejsza wariancję gradientu i wykorzystuje równoległość GPU. Mniejszy batch = więcej szumu, większy = stabilniej, ale zwykle wymaga zwiększenia learning rate (zasada liniowego skalowania + warmup).

**Rozszerzenia:** momentum, Nesterov, Adam/AdamW.

**Praktyka:** mieszaj dane co epokę (`shuffle=True`), przy różnych rozmiarach batcha skaluj lr, stosuj gradient clipping.

```python
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
for xb, yb in loader:
    loss = criterion(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
sched.step()
```

**Źródła:**
- [Outcome School: Math Behind Gradient Descent](https://outcomeschool.com/blog/math-behind-gradient-descent)
- [PyTorch docs: torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html)
- [Optimization Methods for Large-Scale Machine Learning (Bottou, Curtis, Nocedal, arXiv)](https://arxiv.org/abs/1606.04838)
- [scikit-learn: Stochastic Gradient Descent](https://scikit-learn.org/stable/modules/sgd.html)

---

<a id="q63"></a>
### 63. Czym są zanikające gradienty (vanishing gradients)?

**Odpowiedź:**

**Vanishing gradient** to zjawisko, w którym gradienty w warstwach bliższych wejściu stają się bardzo małe (zbliżają się do zera), więc te warstwy uczą się bardzo wolno lub wcale. Odwrotny problem: **exploding gradients** (gradienty rosną wykładniczo).

**Przyczyna:** backpropagation mnoży pochodne warstw łańcuchowo:

$$\frac{\partial L}{\partial W_1}=\frac{\partial L}{\partial a_L}\prod_{l=2}^{L}\frac{\partial a_l}{\partial a_{l-1}}$$

Jeśli czynniki są typowo $<1$ (np. pochodna sigmoidy $\le 0.25$, tanh $\le 1$ ale saturuje), iloczyn maleje wykładniczo z głębokością. W RNN ten sam efekt występuje w czasie (BPTT) - model nie uczy się długich zależności.

**Objawy:** loss stoi w miejscu, wagi początkowych warstw prawie się nie zmieniają, norma gradientów per warstwa maleje do ~0.

**Rozwiązania:**
- **Aktywacje** bez saturacji: **ReLU** i warianty (Leaky ReLU, GELU, SiLU).
- **Inicjalizacja** wag: Xavier/Glorot (tanh/sigmoid), He/Kaiming (ReLU).
- **Normalizacja:** BatchNorm, LayerNorm.
- **Połączenia rezydualne** (ResNet, Transformery) - gradient płynie skrótem.
- **Bramkowane RNN:** LSTM, GRU.
- **Gradient clipping** (przede wszystkim przeciw eksplodującym).
- Mniejsza głębokość, dobre optymalizatory (Adam), pretraining warstw (historycznie).

**Diagnoza:**
```python
for n, p in model.named_parameters():
    if p.grad is not None:
        print(n, p.grad.norm().item())
```

**Źródła:**
- [On the difficulty of training Recurrent Neural Networks (Pascanu et al., arXiv)](https://arxiv.org/abs/1211.5063)
- [Deep Residual Learning for Image Recognition (arXiv)](https://arxiv.org/abs/1512.03385)
- [Delving Deep into Rectifiers (He initialization, arXiv)](https://arxiv.org/abs/1502.01852)
- [Wikipedia: Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

---

<a id="q64"></a>
### 64. Czym jest learning rate? Jak dobrać dobry?

**Odpowiedź:**

**Learning rate** $\eta$ to hiperparametr określający wielkość kroku aktualizacji wag: $\theta\leftarrow\theta-\eta\nabla L$. To zwykle **najważniejszy hiperparametr** w treningu sieci.

**Wpływ:**
- **Za duży:** loss oscyluje lub rośnie/NaN (rozbieżność), przeskakuje minimum.
- **Za mały:** bardzo wolna zbieżność, utknięcie w płaskich obszarach/słabych minimach, marnowanie zasobów.
- **Dobry:** szybki, stabilny spadek loss.

**Jak dobierać:**
1. **LR range test** (Smith): rośnij lr wykładniczo w trakcie krótkiego treningu i wybierz wartość tuż przed miejscem, gdzie loss zaczyna rosnąć (lub ~10x mniejszą od punktu rozbieżności).
2. **Przeszukiwanie w skali logarytmicznej** ($10^{-5}$...$10^{-1}$), random search / Bayesian (Optuna).
3. **Wartości startowe:** Adam/AdamW: $10^{-4}$-$3\cdot10^{-4}$ (fine-tuning Transformerów: $10^{-5}$-$5\cdot10^{-5}$); SGD z momentum: $0.01$-$0.1$.
4. **Harmonogramy (schedules):** step decay, cosine annealing, warmup (liniowy wzrost na początku - kluczowy dla Transformerów), one-cycle, ReduceLROnPlateau.
5. **Skalowanie z batch size:** przy większym batchu zwykle zwiększa się lr (liniowo/pierwiastkowo) z warmupem.
6. **Różne lr dla warstw** (discriminative fine-tuning: mniejszy lr dla pretrenowanych warstw).

**Zależności:** optymalizator (Adam toleruje szerszy zakres), batch size, normalizacja, regularyzacja (weight decay).

**Diagnoza:** obserwuj krzywe loss (train/val), normy gradientów.

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=total_steps)
```

**Źródła:**
- [Learning Rate (Amit Shekhar, źródło z README)](https://www.linkedin.com/posts/amit-shekhar-iitbhu_ml-ai-activity-7438148910260404224-0bxL)
- [Cyclical Learning Rates for Training Neural Networks (Smith, arXiv)](https://arxiv.org/abs/1506.01186)
- [Accurate, Large Minibatch SGD (Goyal et al., arXiv)](https://arxiv.org/abs/1706.02677)
- [PyTorch docs: torch.optim learning rate schedulers](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)

---

<a id="q65"></a>
### 65. Jak learning rate wpływa na trening modelu?

**Odpowiedź:**

Learning rate określa krok po powierzchni straty, więc wpływa na **szybkość, stabilność i jakość końcowego rozwiązania**.

| LR | Zachowanie loss | Skutek |
|---|---|---|
| Zbyt mały | gładki, ale bardzo wolny spadek | długi trening, utknięcie w płaskich obszarach/lokalnych minimach, underfitting w budżecie epok |
| Optymalny | szybki, stabilny spadek | dobra zbieżność |
| Zbyt duży | oscylacje, skoki, wzrost, `NaN` | rozbieżność, brak zbieżności |

**Dodatkowe efekty:**
- **Generalizacja:** stosunkowo duży lr (z szumem SGD) sprzyja płaskim minimom i lepszej generalizacji; końcowe zmniejszenie lr "dopracowuje" rozwiązanie. Stąd harmonogramy: wysoki lr na początku, malejący później.
- **Interakcja z batch size:** większy batch = mniej szumu, zwykle pozwala/wymaga większego lr.
- **Interakcja z optymalizatorem:** Adam skaluje krok per parametr, więc jest mniej wrażliwy niż SGD, ale wciąż wymaga strojenia; przy Adam z lr $10^{-2}$ często niestabilny.
- **Warmup** stabilizuje początek treningu (szczególnie Transformery, gdzie estymaty drugiego momentu są jeszcze niedokładne).
- **Fine-tuning:** zbyt duży lr niszczy pretrenowane cechy (catastrophic forgetting), stąd małe wartości.

**Jak rozpoznać problem na krzywych:**
- Loss rośnie/NaN - lr za duży (lub brak clippingu, zły skalowanie danych).
- Loss spada bardzo wolno, płaskie - lr za mały.
- Zaszumiona krzywa val loss z dużymi wahaniami - zmniejsz lr, zwiększ batch.
- Nagłe skoki podczas treningu - gradient clipping, niższy lr.

**Strategie:** LR range test, schedulery (cosine, one-cycle, ReduceLROnPlateau), warmup, mniejszy lr dla warstw pretrenowanych.

**Źródła:**
- [Learning Rate (Amit Shekhar, źródło z README)](https://www.linkedin.com/posts/amit-shekhar-iitbhu_ml-ai-activity-7438148910260404224-0bxL)
- [Stanford CS231n: Neural Networks Part 3 (learning rate, babysitting the training)](https://cs231n.github.io/neural-networks-3/)
- [Deep Learning book, ch. 8: Optimization for Training Deep Models](https://www.deeplearningbook.org/contents/optimization.html)
- [Cyclical Learning Rates for Training Neural Networks (arXiv)](https://arxiv.org/abs/1506.01186)

---

<a id="q66"></a>
### 66. Jak podejść do strojenia hiperparametrów (hyperparameter tuning)?

**Odpowiedź:**

**Hiperparametry** (learning rate, głębokość drzewa, `C`, liczba warstw, dropout) ustawiamy przed treningiem, w odróżnieniu od parametrów uczonych. Strojenie robimy na **zbiorze walidacyjnym / w CV**, nigdy na teście.

**Metody:**
- **Grid search** - pełna siatka; kosztowna wykładniczo, marnuje budżet na mało istotne wymiary.
- **Random search** - losowe próbki z rozkładów; zwykle znacznie efektywniejszy od siatki (Bergstra & Bengio), bo zwykle tylko kilka hiperparametrów jest ważnych.
- **Optymalizacja bayesowska** (Optuna/TPE, Gaussian Processes, SMAC) - model zastępczy kieruje kolejnymi próbami.
- **Successive halving / Hyperband / ASHA** - wczesne zatrzymywanie słabych konfiguracji, oszczędza obliczenia.
- **Population Based Training** i AutoML dla dużych systemów.

**Proces praktyczny:**
1. Ustal solidny **baseline** i metrykę zgodną z biznesem.
2. Zacznij od hiperparametrów o największym wpływie (learning rate, regularyzacja, rozmiar modelu).
3. Przeszukuj w **skali logarytmicznej** dla lr, `C`, weight decay.
4. Stopniowo zawężaj zakres (coarse-to-fine).
5. Używaj **CV** (lub stałej walidacji dla dużych danych/DL), ustaw seedy, przy DL uśredniaj po kilku seedach dla stabilności.
6. **Early stopping**, pruning nieobiecujących prób.
7. Loguj eksperymenty (MLflow, W&B) i po wyborze trenuj finalny model na train+val, oceniając raz na teście.

**Pułapki:** leakage w CV (preprocessing poza pipeline), overfitting do walidacji przy dużej liczbie prób (nested CV), strojenie na zbyt małym zbiorze walidacyjnym, ignorowanie kosztów obliczeń.

```python
import optuna
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    depth = trial.suggest_int("max_depth", 3, 10)
    return cross_val_score(make_model(lr, depth), X, y, cv=5, scoring="roc_auc").mean()
study = optuna.create_study(direction="maximize"); study.optimize(objective, n_trials=50)
```

**Źródła:**
- [scikit-learn: Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
- [Random Search for Hyper-Parameter Optimization (Bergstra & Bengio, JMLR)](https://jmlr.org/papers/v13/bergstra12a.html)
- [Optuna: dokumentacja](https://optuna.org/)
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization (arXiv)](https://arxiv.org/abs/1603.06560)

---

<a id="q67"></a>
### 67. Czym jest kwantyzacja modelu (model quantization) i kiedy ją stosować?

**Odpowiedź:**

**Kwantyzacja** to zmniejszenie precyzji numerycznej wag (i ewentualnie aktywacji) - np. z FP32/FP16 do INT8, INT4 lub FP8 - aby zmniejszyć rozmiar modelu, zużycie pamięci i przyspieszyć inferencję kosztem niewielkiej utraty dokładności.

**Podstawowy schemat (afiniczny):**
$$q=\text{round}\!\left(\frac{x}{s}\right)+z,\qquad \hat x=s\,(q-z)$$
gdzie $s$ to skala, $z$ - zero-point. Model INT8 zajmuje ok. 4x mniej pamięci niż FP32 (INT4 - ok. 8x).

**Rodzaje:**
- **PTQ (Post-Training Quantization)** - kwantyzacja po treningu, często z małym zbiorem kalibracyjnym (do wyznaczenia zakresów aktywacji). Szybka, ale możliwy spadek jakości. Dla LLM: GPTQ, AWQ, LLM.int8(), GGUF (llama.cpp).
- **QAT (Quantization-Aware Training)** - symulacja kwantyzacji w trakcie treningu (fake quant, STE); lepsza jakość, droższa.
- **Dynamiczna** (aktywacje kwantyzowane w locie), **statyczna** (z kalibracją).
- Granularność: per-tensor, per-channel, per-group.
- **QLoRA** - 4-bitowa baza + adaptery LoRA do fine-tuningu.

**Kiedy stosować:**
- ograniczone zasoby: urządzenia mobilne/edge, tanie GPU/CPU,
- mniejsze opóźnienie i większy throughput, niższy koszt serwowania,
- uruchamianie dużych LLM na jednym GPU (mniejszy ślad pamięci; inferencja LLM jest zwykle ograniczona przepustowością pamięci).

**Trade-offy i pułapki:** spadek jakości (zwykle niewielki dla 8 bitów, większy dla 4 i niżej), wrażliwość outlierów w aktywacjach (LLM - stąd metody z obsługą outlierów), konieczność wsparcia sprzętowego dla realnego przyspieszenia INT8, weryfikacja jakości na własnych zadaniach. Kwantyzacja uzupełnia inne techniki: pruning, destylację.

```python
import torch
qmodel = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

**Źródła:**
- [Outcome School: How Does Model Quantization Work](https://outcomeschool.com/blog/how-does-model-quantization-work)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference (Jacob et al., arXiv)](https://arxiv.org/abs/1712.05877)
- [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale (arXiv)](https://arxiv.org/abs/2208.07339)
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers (arXiv)](https://arxiv.org/abs/2210.17323)
- [QLoRA: Efficient Finetuning of Quantized LLMs (arXiv)](https://arxiv.org/abs/2305.14314)

---

<a id="q68"></a>
### 68. Jak zapewnić sprawiedliwość (fairness) i ograniczyć bias w modelach ML?

**Odpowiedź:**

**Bias** w ML to systematyczne, niesprawiedliwe różnice w wynikach dla grup (np. płeć, wiek, pochodzenie). Źródła: **dane** (niereprezentatywne próbkowanie, historyczne uprzedzenia w etykietach, proxy dla cech wrażliwych jak kod pocztowy), **cechy i etykiety** (błędy pomiaru), **model/cel** (optymalizacja średniej kosztem mniejszości), **wdrożenie** (pętle sprzężenia zwrotnego).

**Metryki fairness** (nie wszystkie da się spełnić naraz - twierdzenia o niemożliwości):
- **Demographic parity** - równy odsetek pozytywnych decyzji w grupach.
- **Equalized odds / equal opportunity** - równe TPR (i FPR) w grupach.
- **Predictive parity / kalibracja** w grupach.
- Analiza metryk **per podgrupa** (slice-based evaluation), także przecięcia (intersectional).

**Działania w trzech fazach:**
1. **Pre-processing:** audyt danych, zrównoważenie reprezentacji (resampling, reweighting), poprawa etykiet, usunięcie/zmiana proxy, dokumentacja (datasheets).
2. **In-processing:** ograniczenia fairness w funkcji celu, regularyzacja, adversarial debiasing, reductions (Fairlearn).
3. **Post-processing:** różne progi decyzyjne per grupa, kalibracja (np. Hardt et al.).

**Proces:** określ, co znaczy "sprawiedliwie" w danym kontekście (z interesariuszami i prawnikami), zmierz na danych walidacyjnych z atrybutami wrażliwymi, dokumentuj (**model cards**), testuj przed wdrożeniem i monitoruj w produkcji (drift, zmiana metryk per grupa), zapewnij ludzki nadzór i wyjaśnialność (SHAP), procedury odwołań.

**Uwagi:** samo usunięcie cechy wrażliwej ("fairness through unawareness") nie wystarcza - proxy nadal przenoszą informację; fairness często wiąże się z kompromisem z dokładnością; wymogi regulacyjne (np. EU AI Act, RODO).

```python
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import recall_score
mf = MetricFrame(metrics={"recall": recall_score, "sel_rate": selection_rate},
                 y_true=y_test, y_pred=y_pred, sensitive_features=A_test)
print(mf.by_group)
```

**Źródła:**
- [A Survey on Bias and Fairness in Machine Learning (Mehrabi et al., arXiv)](https://arxiv.org/abs/1908.09635)
- [Equality of Opportunity in Supervised Learning (Hardt et al., arXiv)](https://arxiv.org/abs/1610.02413)
- [Model Cards for Model Reporting (Mitchell et al., arXiv)](https://arxiv.org/abs/1810.03993)
- [Fairlearn: dokumentacja](https://fairlearn.org/)

---

<a id="q69"></a>
### 69. Czym różnią się Grid Search, Random Search i optymalizacja bayesowska (Bayesian Optimization)?

**Odpowiedź:**

Wszystkie trzy metody służą do strojenia hiperparametrów (learning rate, głębokość drzewa, siła regularyzacji itd.), czyli parametrów, których model nie uczy się z danych przez gradient. Różnią się sposobem wyboru kolejnych punktów do sprawdzenia.

#### Grid Search
- Definiujemy siatkę wartości dla każdego hiperparametru i sprawdzamy **wszystkie kombinacje** (iloczyn kartezjański).
- Zalety: prosta, deterministyczna, w pełni powtarzalna, łatwa do zrównoleglenia.
- Wady: liczba kombinacji rośnie wykładniczo z liczbą wymiarów (przekleństwo wymiarowości). Przy 5 hiperparametrach po 10 wartości daje to 100 000 uczeń modelu. Marnuje budżet na wymiary, które mało wpływają na wynik, bo każdą wartość ważnego parametru testuje tylko tyle razy, ile jest kombinacji pozostałych.

#### Random Search
- Losujemy `n` konfiguracji z zadanych rozkładów (np. learning rate z rozkładu log-jednostajnego).
- Bergstra i Bengio (2012) pokazali, że przy zwykle niskim „efektywnym wymiarze” problemu (tylko kilka hiperparametrów naprawdę ma znaczenie) losowe przeszukiwanie jest wydajniejsze niż siatka: przy tym samym budżecie testuje `n` różnych wartości każdego ważnego parametru, a nie tylko `n^(1/d)`.
- Zalety: budżet jest dowolny (można przerwać w każdej chwili), obsługuje parametry ciągłe i dyskretne, łatwo się zrównolegla.
- Wady: próbki są niezależne, więc nie wykorzystuje informacji z poprzednich ewaluacji.

#### Optymalizacja bayesowska
- Buduje **model zastępczy** (surrogate) funkcji celu `f(λ)` (wynik walidacyjny w zależności od hiperparametrów) na podstawie dotychczasowych obserwacji: najczęściej proces gaussowski (GP), albo estymator TPE lub lasy losowe (SMAC).
- **Funkcja akwizycji** (acquisition function), np. Expected Improvement, UCB, decyduje, gdzie ewaluować następny punkt, równoważąc eksplorację (niepewne obszary) i eksploatację (obszary z dobrym przewidywanym wynikiem).
- Zalety: wymaga najmniej ewaluacji, co jest kluczowe, gdy jedno trenowanie trwa godziny.
- Wady: z natury sekwencyjna (równoległość wymaga dodatkowych trików), narzut obliczeniowy modelu zastępczego, GP słabo skaluje się powyżej kilkudziesięciu wymiarów, gorzej radzi sobie z parametrami warunkowymi i kategorycznymi.

| Cecha | Grid | Random | Bayesian |
|---|---|---|---|
| Wykorzystuje historię | nie | nie | tak |
| Skalowanie z wymiarem | bardzo złe | dobre | umiarkowane |
| Równoległość | trywialna | trywialna | trudniejsza |
| Koszt narzutu | brak | brak | średni |
| Typowe użycie | 1-3 parametry | punkt startowy, wiele parametrów | drogie modele |

#### Praktyczne wskazówki
- Skalę logarytmiczną stosuj dla learning rate i regularyzacji.
- Dobrym schematem jest najpierw szeroki random search, potem zawężenie zakresu (lub optymalizacja bayesowska).
- Łącz z wczesnym zatrzymywaniem (Hyperband/ASHA, pruning w Optuna).
- Zawsze oceniaj na zbiorze walidacyjnym lub w walidacji krzyżowej, a końcowy wynik raportuj na osobnym zbiorze testowym, aby uniknąć przecieku.

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import loguniform
rs = RandomizedSearchCV(model, {"C": loguniform(1e-3, 1e3)}, n_iter=50, cv=5)
```

**Źródła:**
- [Bergstra, Bengio - Random Search for Hyper-Parameter Optimization (JMLR)](https://jmlr.org/papers/v13/bergstra12a.html)
- [Snoek, Larochelle, Adams - Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944)
- [scikit-learn - Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)

---

<a id="q70"></a>
### 70. Wyjaśnij optymalizację hiperparametrów metodą TPE (Tree-structured Parzen Estimator).

**Odpowiedź:**

TPE to algorytm optymalizacji bayesowskiej zaproponowany przez Bergstrę i in. (2011), używany domyślnie w bibliotekach **Hyperopt** i **Optuna**. Zamiast modelować `p(y | x)` (jak proces gaussowski), modeluje **`p(x | y)`** i `p(y)`, a następnie wykorzystuje regułę Bayesa.

#### Mechanizm
1. Zbieramy dotychczasowe obserwacje `(x_i, y_i)` (x - konfiguracja hiperparametrów, y - loss, mniejszy jest lepszy).
2. Ustalamy próg `y*` jako kwantyl `γ` obserwowanych wartości (zwykle `γ ≈ 0.15-0.25`, czyli „najlepsze” obserwacje).
3. Dzielimy obserwacje na dwie grupy i estymujemy dla każdej gęstość:
   - `l(x) = p(x | y < y*)` - gęstość konfiguracji „dobrych”,
   - `g(x) = p(x | y ≥ y*)` - gęstość konfiguracji „złych”.
4. Estymatory to **Parzen estimators** (kernel density estimators, mieszaniny rozkładów, np. gaussowskich dla parametrów ciągłych i kategorycznych dla dyskretnych).
5. Można pokazać, że Expected Improvement jest proporcjonalne do

$$EI(x) \propto \left(\gamma + \frac{g(x)}{l(x)}(1-\gamma)\right)^{-1}$$

   więc maksymalizacja EI to maksymalizacja stosunku **`l(x)/g(x)`**.
6. Losujemy wiele kandydatów z `l(x)`, wybieramy tego z największym `l(x)/g(x)`, ewaluujemy go i powtarzamy.

#### Dlaczego „tree-structured”
Przestrzeń poszukiwań może mieć strukturę drzewiastą (parametry warunkowe): np. jeśli `optimizer = "sgd"`, to istnieje `momentum`, a jeśli `"adam"`, to `beta2`. TPE modeluje takie zależności naturalnie, ponieważ estymuje gęstości dla każdego wymiaru (w wersji podstawowej niezależnie) i uwzględnia tylko aktywne parametry.

#### Zalety i wady
- (+) Dobrze skaluje się do wielu wymiarów, obsługuje parametry kategoryczne, warunkowe i całkowite.
- (+) Koszt jest liniowy względem liczby obserwacji, znacznie tańszy niż GP (sześcienny).
- (+) Łatwo łączy się z pruningiem (Optuna).
- (-) Klasyczny TPE ignoruje interakcje między parametrami (niezależne gęstości na wymiar), choć Optuna oferuje tryb `multivariate=True`.
- (-) Wyniki zależą od `γ` i liczby losowych startowych prób.

```python
import optuna
def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    depth = trial.suggest_int("depth", 3, 10)
    return train_and_eval(lr, depth)
study = optuna.create_study(direction="minimize",
                            sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, n_trials=100)
```

**Źródła:**
- [Bergstra et al. - Algorithms for Hyper-Parameter Optimization (NeurIPS 2011)](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html)
- [Optuna - TPESampler documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)
- [Akiba et al. - Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)
- [Hyperopt - documentation](https://hyperopt.github.io/hyperopt/)

---

<a id="q71"></a>
### 71. Wyjaśnij optymalizację bayesowską (Bayesian Optimization).

**Odpowiedź:**

Optymalizacja bayesowska to metoda **globalnej optymalizacji drogich funkcji typu black-box** `f(x)`, dla których nie mamy gradientu, a każda ewaluacja jest kosztowna (np. pełny trening sieci). Cel: `x* = argmin f(x)` przy jak najmniejszej liczbie ewaluacji.

#### Dwa składniki
1. **Model zastępczy (surrogate)** - probabilistyczny model `f`, dający dla każdego `x` średnią `μ(x)` i niepewność `σ(x)`. Najpopularniejszy: **proces gaussowski** (GP) z jądrem np. Matérn lub RBF. Alternatywy: TPE, lasy losowe (SMAC), sieci bayesowskie.
2. **Funkcja akwizycji** `α(x)` - tania do optymalizacji reguła wybierająca kolejny punkt. Przykłady:
   - Probability of Improvement (PI),
   - **Expected Improvement (EI)**: $EI(x) = \mathbb{E}[\max(0, f_{best} - f(x))]$,
   - **Upper/Lower Confidence Bound**: $LCB(x) = \mu(x) - \kappa\,\sigma(x)$, gdzie `κ` steruje eksploracją,
   - Thompson sampling, Knowledge Gradient.

#### Pętla algorytmu
1. Wylosuj kilka punktów początkowych i ewaluuj `f`.
2. Dopasuj surrogate do wszystkich dotychczasowych danych.
3. Zmaksymalizuj `α(x)` (np. L-BFGS z wieloma startami) i wybierz `x_next`.
4. Ewaluuj `f(x_next)`, dodaj do danych.
5. Powtarzaj do wyczerpania budżetu.

#### Eksploracja vs eksploatacja
Funkcja akwizycji faworyzuje punkty o niskiej przewidywanej wartości (eksploatacja) oraz punkty o dużej niepewności (eksploracja). Dzięki temu metoda unika utknięcia w lokalnym minimum i nie marnuje ewaluacji na obszary już dobrze poznane.

#### Zastosowania i ograniczenia
- Strojenie hiperparametrów, projektowanie architektur (NAS), eksperymenty fizyczne i chemiczne, A/B testy.
- GP ma koszt `O(n^3)` względem liczby obserwacji i słabo działa w bardzo wysokich wymiarach (>20-50). Zakłada gładkość funkcji.
- Obserwacje szumowe modeluje się dodatkowym członem szumu w GP.
- Wersje rozszerzone: batch BO (równoległość), multi-fidelity (BOHB), ograniczenia, optymalizacja wielokryterialna.

Narzędzia: `scikit-optimize`, `BoTorch`, `Ax`, `Optuna`, `SMAC`, `bayes_opt`.

**Źródła:**
- [Frazier - A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)
- [Snoek et al. - Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944)
- [BoTorch - documentation](https://botorch.org/)
- [Wikipedia - Bayesian optimization](https://en.wikipedia.org/wiki/Bayesian_optimization)

---

<a id="q72"></a>
### 72. Wyjaśnij optymalizator Adam.

**Odpowiedź:**

**Adam** (Adaptive Moment Estimation, Kingma i Ba, 2014) łączy dwie idee: **momentum** (wykładnicza średnia gradientów) oraz **adaptacyjne learning rate per parametr** w stylu RMSprop (wykładnicza średnia kwadratów gradientów).

#### Algorytm
Dla gradientu `g_t` w kroku `t`:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

Korekta obciążenia (bias correction), bo `m_0 = v_0 = 0`:

$$\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}$$

Aktualizacja:

$$\theta_t = \theta_{t-1} - \alpha \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}$$

Domyślne wartości: `α = 1e-3`, `β1 = 0.9`, `β2 = 0.999`, `ε = 1e-8`.

#### Intuicja
- `m_t` to wygładzony kierunek (redukuje szum gradientu stochastycznego).
- `sqrt(v_t)` szacuje skalę gradientu danego parametru: parametry z dużymi gradientami dostają mniejszy krok, z małymi - większy. Efektywny krok jest w przybliżeniu ograniczony przez `α`, niezależnie od skali gradientu.
- Bias correction jest ważna na początku treningu, gdy średnie są jeszcze zaniżone.

#### Zalety
- Dobrze działa „out of the box”, mało wrażliwy na dobór `α`.
- Radzi sobie z rzadkimi gradientami i niestacjonarnymi celami.
- Standard w Transformerach, GAN-ach, wielu zadaniach NLP i vision.

#### Wady i pułapki
- Może generalizować gorzej niż SGD z momentum w niektórych zadaniach (np. klasyfikacja obrazów), choć wyniki zależą od strojenia.
- Klasyczny Adam z L2 w gradiencie nie realizuje prawdziwego weight decay. Poprawka to **AdamW** (Loshchilov i Hutter), w którym decay jest rozdzielony od adaptacji: `θ ← θ - α(... ) - αλθ`.
- Wymaga 2 dodatkowych buforów na parametr (pamięć 3x względem samych wag), co ma znaczenie przy dużych LLM.
- W treningu Transformerów zwykle stosuje się warmup i schedule (cosine), a często `β2 = 0.95-0.98` i `ε` większe dla stabilności.

```python
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01)
```

**Źródła:**
- [Kingma, Ba - Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- [Loshchilov, Hutter - Decoupled Weight Decay Regularization (AdamW)](https://arxiv.org/abs/1711.05101)
- [PyTorch - torch.optim.Adam](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
- [Ruder - An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)

---

<a id="q73"></a>
### 73. Wyjaśnij optymalizator RMSprop.

**Odpowiedź:**

**RMSprop** (Root Mean Square Propagation) zaproponował Geoffrey Hinton na slajdach do kursu Coursera (lekcja 6e), bez formalnej publikacji. Powstał jako poprawka Adagrada, w którym learning rate maleje zbyt agresywnie.

#### Problem, który rozwiązuje
Wartość `learning rate` powinna dostosowywać się do skali gradientu każdego parametru: dla parametrów z dużymi, oscylującymi gradientami krok powinien być mniejszy. Adagrad kumuluje sumę **wszystkich** kwadratów gradientów, więc mianownik rośnie monotonicznie, a uczenie z czasem zamiera. RMSprop zastępuje sumę **wykładniczą średnią kroczącą**, która „zapomina” stare gradienty.

#### Algorytm

$$v_t = \rho\, v_{t-1} + (1-\rho)\, g_t^2$$
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon}\, g_t$$

Typowe wartości: `ρ (alpha w PyTorch) = 0.9` (0.99 w domyślnym PyTorch), `η = 1e-3`, `ε = 1e-8`.

#### Intuicja
Dzielenie przez `sqrt(v_t)` (RMS gradientów) normalizuje krok: parametry o dużych gradientach są tłumione, o małych - wzmacniane. W kierunkach z oscylacjami (np. „wąska dolina”) zmniejsza to zygzakowanie i pozwala użyć większego `η`. Efekt jest zbliżony do normalizacji gradientu przez jego niedawną wielkość.

#### Warianty
- **Centered RMSprop** - dzieli przez estymatę odchylenia standardowego (odejmuje kwadrat średniej gradientu).
- **RMSprop z momentum** - dodatkowy bufor prędkości.

#### Porównanie
| Optymalizator | Momentum | Adaptacja per parametr | Bias correction |
|---|---|---|---|
| SGD+momentum | tak | nie | nie |
| Adagrad | nie | tak (suma) | nie |
| RMSprop | opcjonalnie | tak (średnia wykładnicza) | nie |
| Adam | tak | tak | tak |

Adam można rozumieć jako RMSprop + momentum + bias correction.

#### Kiedy używać
Dobry wybór dla sieci rekurencyjnych (RNN) i uczenia ze wzmocnieniem (np. oryginalny DQN), przy niestacjonarnych celach. Współcześnie częściej wybiera się Adam/AdamW.

```python
opt = torch.optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.99, eps=1e-8, momentum=0.0)
```

**Źródła:**
- [Hinton - Neural Networks for Machine Learning, Lecture 6 slides (RMSprop)](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
- [PyTorch - torch.optim.RMSprop](https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)
- [Ruder - An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)

---

<a id="q74"></a>
### 74. Czym jest optymalizator Adagrad?

**Odpowiedź:**

**Adagrad** (Adaptive Gradient, Duchi, Hazan, Singer, 2011) to optymalizator, który **dostosowuje learning rate osobno dla każdego parametru** na podstawie historii gradientów: parametry często aktualizowane dostają mniejsze kroki, a rzadko aktualizowane - większe.

#### Algorytm
Akumulacja kwadratów gradientów od początku treningu:

$$G_t = G_{t-1} + g_t^2$$
$$\theta_{t} = \theta_{t-1} - \frac{\eta}{\sqrt{G_t} + \epsilon}\, g_t$$

(operacje element-wise; w wersji pełnej macierzowej używa się macierzy zewnętrznych iloczynów gradientów, ale jest ona niepraktyczna).

#### Intuicja
- Parametr z dużymi historycznymi gradientami ma duże `G_t`, więc jego efektywny learning rate szybko maleje.
- Parametr z rzadkimi (sparse) gradientami zachowuje mały `G_t` i uczy się szybciej, gdy w końcu pojawi się informacja. Dlatego Adagrad świetnie sprawdza się przy **rzadkich cechach**: embeddingi słów (np. GloVe), modele liniowe dla reklam i rekomendacji (bag-of-words, one-hot).
- Nie wymaga ręcznego strojenia schedule'u `η` (zwykle 0.01).

#### Wady
- `G_t` rośnie monotonicznie, więc efektywny learning rate **dąży do zera**, a uczenie zwalnia lub zatrzymuje się przedwcześnie, szczególnie w głębokich sieciach i problemach niewypukłych.
- To właśnie ta wada zmotywowała **Adadelta** i **RMSprop** (wykładnicza średnia zamiast sumy) oraz Adam.

#### Kiedy stosować
- Dane rzadkie, problemy wypukłe, krótsze treningi.
- Rzadko do głębokich sieci, gdzie lepsze są RMSprop/Adam/AdamW.

```python
opt = torch.optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, eps=1e-10)
```

Teoretycznie Adagrad ma gwarancje regret `O(√T)` w optymalizacji wypukłej online i jest adaptacyjny względem geometrii danych.

**Źródła:**
- [Duchi, Hazan, Singer - Adaptive Subgradient Methods for Online Learning and Stochastic Optimization (JMLR)](https://jmlr.org/papers/v12/duchi11a.html)
- [PyTorch - torch.optim.Adagrad](https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html)
- [Ruder - An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)

---

## Deep Learning

<a id="q75"></a>
### 75. Czym są sieci neuronowe (neural networks)?

**Odpowiedź:**

**Sieć neuronowa** to parametryczny model złożony z warstw prostych jednostek (neuronów), który uczy się aproksymować funkcję `f: X → Y` na podstawie danych. Inspiracja biologiczna jest luźna; współcześnie to po prostu złożenie funkcji różniczkowalnych trenowane gradientowo.

#### Neuron
Pojedynczy neuron oblicza ważoną sumę wejść i przepuszcza ją przez nieliniową **funkcję aktywacji** `φ`:

$$y = \varphi\left(\sum_i w_i x_i + b\right) = \varphi(w^\top x + b)$$

#### Warstwy
Warstwa w pełni połączona (dense): $h = \varphi(Wx + b)$. Sieć to złożenie warstw:

$$f(x) = f_L(\dots f_2(f_1(x)))$$

- **warstwa wejściowa** - cechy,
- **warstwy ukryte** - uczą się coraz bardziej abstrakcyjnych reprezentacji,
- **warstwa wyjściowa** - predykcja (softmax dla klasyfikacji, liniowa dla regresji, sigmoid dla binarnej).

#### Trening
1. **Forward pass** - obliczenie predykcji.
2. **Funkcja straty** (loss) - np. cross-entropy, MSE.
3. **Backpropagation** - gradienty straty względem parametrów przez regułę łańcuchową.
4. **Aktualizacja** parametrów optymalizatorem (SGD, Adam) na mini-batchach.

#### Dlaczego działają
- **Twierdzenie o uniwersalnej aproksymacji**: sieć z jedną warstwą ukrytą i nieliniową aktywacją może aproksymować dowolną ciągłą funkcję na zbiorze zwartym z dowolną dokładnością (przy wystarczającej szerokości). Nie mówi ono jednak nic o tym, jak łatwo taką sieć wytrenować ani o generalizacji.
- Głębokość pozwala reprezentować funkcje hierarchiczne wydajniej (mniej parametrów niż płytka, bardzo szeroka sieć).

#### Popularne architektury
MLP (feedforward), CNN (obrazy), RNN/LSTM (sekwencje), Transformer (tekst, obraz, multimodalne), autoenkodery, GAN, modele dyfuzyjne, GNN.

#### Uwagi praktyczne
Sieci wymagają dużo danych i mocy obliczeniowej, są podatne na overfitting (regularyzacja: dropout, weight decay, augmentacja, early stopping) i są trudniejsze w interpretacji niż modele klasyczne.

```python
import torch.nn as nn
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
```

**Źródła:**
- [Goodfellow, Bengio, Courville - Deep Learning (książka)](https://www.deeplearningbook.org/)
- [Nielsen - Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [CS231n - Neural Networks Part 1: Setting up the Architecture](https://cs231n.github.io/neural-networks-1/)
- [Wikipedia - Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)

---

<a id="q76"></a>
### 76. Wyjaśnij sieć neuronową typu feedforward (Feedforward Neural Network).

**Odpowiedź:**

**Feedforward Neural Network (FNN)**, zwana też perceptronem wielowarstwowym (MLP), to sieć, w której informacja płynie w jednym kierunku: od wejścia przez warstwy ukryte do wyjścia, **bez cykli i sprzężeń zwrotnych**. Graf obliczeń jest acykliczny (DAG).

#### Budowa
Dla warstwy `l`:

$$h^{(l)} = \varphi\left(W^{(l)} h^{(l-1)} + b^{(l)}\right), \qquad h^{(0)} = x$$

- `W^{(l)}` - macierz wag o wymiarach `(n_l × n_{l-1})`,
- `b^{(l)}` - wektor biasów,
- `φ` - nieliniowość (ReLU, GELU, tanh).

Wyjście: `ŷ = g(W^{(L)} h^{(L-1)} + b^{(L)})`, gdzie `g` to softmax/sigmoid/tożsamość zależnie od zadania.

#### Dlaczego nieliniowość jest niezbędna
Bez `φ` złożenie warstw liniowych to nadal jedna transformacja liniowa `W_L ... W_1 x`, więc głębokość nic nie daje.

#### Przykład
Sieć 2-3-1 na danych 2D: `h = ReLU(W1 x + b1)` (3 neurony), `ŷ = σ(w2^T h + b2)`. Liczba parametrów: `2·3 + 3 + 3·1 + 1 = 13`.

#### Trening
Funkcja straty (np. cross-entropy) + backpropagation + optymalizator na mini-batchach. Inicjalizacja Xaviera/He zapobiega zanikaniu i eksplozji sygnału.

#### FFN w Transformerach
Każdy blok Transformera zawiera **position-wise feed-forward network**, stosowaną niezależnie do każdego tokenu:

$$FFN(x) = W_2\,\varphi(W_1 x + b_1) + b_2$$

Wymiar wewnętrzny to zwykle 4x wymiar modelu (w wariantach z bramką, np. SwiGLU, inaczej). Warstwy FFN zawierają większość parametrów Transformera i, według badań interpretowalności, pełnią rolę pamięci wiedzy (key-value), podczas gdy attention miesza informacje między tokenami. W Mixture of Experts FFN jest zastępowana zbiorem ekspertów.

#### Zastosowania i ograniczenia
- Dane tabelaryczne, klasyfikacja, regresja, komponent większych architektur.
- Nie wykorzystuje struktury danych (przestrzennej - CNN, sekwencyjnej - RNN/Transformer), więc dla obrazów ma zbyt wiele parametrów i słabą generalizację.

```python
class MLP(nn.Module):
    def __init__(self, d_in, d_h, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_h), nn.ReLU(), nn.Linear(d_h, d_out))
    def forward(self, x): return self.net(x)
```

**Źródła:**
- [Outcome School - Feed-Forward Networks in LLMs](https://outcomeschool.com/blog/feed-forward-networks-in-llms)
- [Deep Learning book - Chapter 6: Deep Feedforward Networks](https://www.deeplearningbook.org/contents/mlp.html)
- [Vaswani et al. - Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Geva et al. - Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)

---

<a id="q77"></a>
### 77. Czym są propagacja w przód (forward propagation) i propagacja wsteczna (backward propagation)?

**Odpowiedź:**

Trening sieci neuronowej to powtarzanie cyklu: **forward pass** (obliczenie predykcji i straty), **backward pass** (obliczenie gradientów), **aktualizacja** parametrów.

#### Forward propagation
Dane przechodzą przez sieć warstwa po warstwie:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = \varphi(z^{(l)})$$

Na końcu liczymy stratę `L(ŷ, y)`. W forward zapamiętywane są aktywacje pośrednie `a^{(l)}` i `z^{(l)}`, bo będą potrzebne w backward pass (stąd zużycie pamięci proporcjonalne do głębokości i batch size; technika **gradient checkpointing** zamienia pamięć na dodatkowe obliczenia).

#### Backward propagation
Obliczamy gradient straty względem każdego parametru za pomocą **reguły łańcuchowej**, idąc od wyjścia do wejścia. Definiujemy błąd warstwy $\delta^{(l)} = \partial L / \partial z^{(l)}$:

$$\delta^{(L)} = \nabla_{a^{(L)}} L \odot \varphi'(z^{(L)})$$
$$\delta^{(l)} = \left(W^{(l+1)\top}\delta^{(l+1)}\right)\odot \varphi'(z^{(l)})$$
$$\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} a^{(l-1)\top}, \qquad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)}$$

(dla softmax + cross-entropy `δ^{(L)} = ŷ - y`).

#### Aktualizacja
Optymalizator używa gradientów: `θ ← θ - η ∇θ L` (lub Adam itd.).

#### Kluczowe różnice
| | Forward | Backward |
|---|---|---|
| Kierunek | wejście -> wyjście | wyjście -> wejście |
| Wynik | predykcja, strata | gradienty |
| Potrzebuje | wag | aktywacji z forward + wag |
| Koszt | ok. 1x | ok. 2x forward |

#### Uwagi praktyczne
- W inferencji wykonujemy tylko forward (`torch.no_grad()`), co oszczędza pamięć.
- Zanikanie/eksplozja gradientów wynika z iloczynu wielu czynników `W^T φ'` w backward (rozwiązania: ReLU, inicjalizacja He, normalizacje, połączenia rezydualne, gradient clipping).

```python
loss = criterion(model(x), y)   # forward
opt.zero_grad()
loss.backward()                  # backward
opt.step()                       # update
```

**Źródła:**
- [Outcome School - Math Behind Backpropagation](https://outcomeschool.com/blog/math-behind-backpropagation)
- [Nielsen - How the backpropagation algorithm works](http://neuralnetworksanddeeplearning.com/chap2.html)
- [PyTorch - Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html)

---

<a id="q78"></a>
### 78. Czym jest propagacja wsteczna (backpropagation)?

**Odpowiedź:**

**Backpropagation** to algorytm wydajnego obliczania gradientu funkcji straty względem wszystkich parametrów sieci. Jest szczególnym przypadkiem **automatycznego różniczkowania w trybie wstecznym** (reverse-mode autodiff) i opiera się na **regule łańcuchowej**. Popularyzowali go Rumelhart, Hinton i Williams (1986).

#### Idea
Strata to złożenie funkcji: $L = \ell(f_L(\dots f_1(x;\theta_1)\dots;\theta_L), y)$. Reguła łańcuchowa:

$$\frac{\partial L}{\partial \theta_l} = \frac{\partial L}{\partial a_L}\cdot\frac{\partial a_L}{\partial a_{L-1}}\cdots\frac{\partial a_{l}}{\partial \theta_l}$$

Zamiast liczyć każdy gradient od zera, propagujemy wstecz gradient „po wyjściu” (**upstream gradient**) i w każdym węźle mnożymy go przez lokalną pochodną. Dzięki ponownemu użyciu wyników pośrednich koszt jest rzędu kosztu jednego forward pass, a nie proporcjonalny do liczby parametrów (jak w różnicach numerycznych).

#### Mały przykład
`y = σ(wx + b)`, `L = (y - t)^2`, `x=1, w=0.5, b=0, t=1`:
- `z = 0.5`, `y = σ(0.5) ≈ 0.622`
- `∂L/∂y = 2(y - t) ≈ -0.756`
- `∂y/∂z = y(1 - y) ≈ 0.235`
- `∂L/∂w = ∂L/∂y · ∂y/∂z · x ≈ -0.756 · 0.235 · 1 ≈ -0.178`

Gradient ujemny oznacza, że zwiększenie `w` zmniejszy stratę.

#### Kroki
1. Forward pass z zapamiętaniem wartości pośrednich.
2. Obliczenie gradientu straty względem wyjścia.
3. Dla warstw od ostatniej do pierwszej: obliczenie gradientu wag i gradientu względem wejścia warstwy.
4. Aktualizacja parametrów optymalizatorem.

#### Problemy
- **Vanishing/exploding gradients** - iloczyn wielu czynników mniejszych/większych od 1. Leczą je ReLU, inicjalizacja He/Xavier, BatchNorm/LayerNorm, skip connections, LSTM, gradient clipping.
- **Pamięć** - wymaga przechowania aktywacji (checkpointing).
- W RNN stosuje się **BPTT** (backpropagation through time).
- Funkcje niedifferentiowalne (np. argmax) wymagają aproksymacji (straight-through estimator, Gumbel-softmax).

#### Kontrola poprawności
**Gradient checking**: porównanie z gradientem numerycznym `(L(θ+ε) - L(θ-ε)) / 2ε`.

W praktyce nikt nie implementuje backprop ręcznie: PyTorch/TensorFlow/JAX budują graf obliczeniowy i robią to automatycznie (`loss.backward()`).

**Źródła:**
- [Outcome School - Math Behind Backpropagation](https://outcomeschool.com/blog/math-behind-backpropagation)
- [Rumelhart, Hinton, Williams - Learning representations by back-propagating errors (Nature)](https://www.nature.com/articles/323533a0)
- [CS231n - Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/)
- [Baydin et al. - Automatic differentiation in machine learning: a survey](https://arxiv.org/abs/1502.05767)

---

<a id="q79"></a>
### 79. Wymień i wyjaśnij kilka hiperparametrów używanych do trenowania sieci neuronowej.

**Odpowiedź:**

**Hiperparametry** to ustawienia wybierane przed treningiem (nie uczone przez gradient), wpływające na proces uczenia, pojemność modelu i generalizację.

#### Optymalizacja
- **Learning rate (η)** - najważniejszy hiperparametr. Za duży: rozbieżność lub oscylacje; za mały: wolna zbieżność, ryzyko utknięcia. Stosuje się **schedule** (cosine, step decay, warmup + decay, one-cycle) i szuka na skali logarytmicznej.
- **Optymalizator** i jego parametry: SGD (momentum 0.9), Adam (`β1, β2, ε`), AdamW.
- **Batch size** - mały: bardziej zaszumiony gradient (działa jak regularyzacja), mniejsze zużycie pamięci; duży: stabilniejszy gradient, lepsze wykorzystanie GPU, ale często wymaga skalowania `η` i może pogarszać generalizację.
- **Liczba epok** / kroków - kontrolowana zwykle przez early stopping.
- **Gradient clipping** - limit normy gradientu (ważne dla RNN i Transformerów).

#### Architektura
- Liczba warstw (głębokość) i neuronów (szerokość) - pojemność modelu.
- **Funkcja aktywacji** (ReLU, GELU...).
- Rozmiar filtrów, stride, padding w CNN; liczba głów attention w Transformerze.
- **Inicjalizacja** wag (He, Xavier).

#### Regularyzacja
- **Dropout rate** (np. 0.1-0.5),
- **Weight decay / L2** (np. 1e-4 - 1e-2),
- Siła augmentacji danych, label smoothing,
- Parametry normalizacji (momentum BN).

#### Funkcja straty
Wybór funkcji straty, wagi klas, parametr focal loss `γ`.

#### Jak je dobierać
1. Zacznij od sprawdzonych ustawień (np. AdamW, `lr=3e-4`), dopasuj `η` jako pierwsze.
2. Użyj **random search** lub optymalizacji bayesowskiej (Optuna) zamiast siatki.
3. Oceniaj na zbiorze walidacyjnym, testuj raz na końcu.
4. Diagnozuj krzywe uczenia: gdy train loss jest wysoki - zwiększ pojemność/trenuj dłużej; gdy walidacja odstaje - regularyzuj.
5. Sanity check: overfit na małym podzbiorze.

Nie mylić z **parametrami** (wagi `W`, biasy `b`), które są uczone z danych.

**Źródła:**
- [Bengio - Practical Recommendations for Gradient-Based Training of Deep Architectures](https://arxiv.org/abs/1206.5533)
- [Karpathy - A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)
- [Google Research - Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
- [Deep Learning book - Chapter 11: Practical Methodology](https://www.deeplearningbook.org/contents/guidelines.html)

---

<a id="q80"></a>
### 80. Jaka jest przewaga głębokiego uczenia nad tradycyjnym uczeniem maszynowym?

**Odpowiedź:**

Głębokie uczenie (deep learning, DL) nie jest zawsze lepsze, ale ma kluczowe zalety w określonych klasach problemów.

#### Główne zalety
1. **Automatyczne uczenie cech (representation learning)**: w klasycznym ML inżynier ręcznie tworzy cechy (SIFT/HOG dla obrazów, TF-IDF dla tekstu). Sieć uczy hierarchię reprezentacji end-to-end: krawędzie -> tekstury -> części -> obiekty.
2. **Praca na danych niestrukturalnych**: obrazy, dźwięk, tekst, wideo, grafy. Tu DL zdominowało dziedzinę (CNN, Transformery).
3. **Skalowanie z danymi i mocą obliczeniową**: klasyczne algorytmy zwykle osiągają plateau, a jakość sieci wciąż rośnie wraz z danymi, parametrami i obliczeniami (scaling laws).
4. **Transfer learning i modele bazowe**: modele pretrenowane (BERT, ResNet, LLM) można dostroić do zadania na małej ilości danych.
5. **Elastyczność architektur i multimodalność**: jeden framework dla tekstu, obrazu i dźwięku; uczenie end-to-end z różnymi celami (generacja, detekcja, RL).
6. **Wysoka pojemność**: aproksymacja złożonych nieliniowych zależności.

#### Wady i kiedy wybrać klasyczne ML
| Aspekt | Deep Learning | Klasyczne ML |
|---|---|---|
| Ilość danych | duża | mała/średnia wystarcza |
| Dane tabelaryczne | często gorsze | zwykle lepsze (XGBoost, LightGBM) |
| Interpretowalność | niska | wyższa (liniowe, drzewa) |
| Koszt obliczeń | wysoki (GPU) | niski |
| Inżynieria cech | minimalna | kluczowa |
| Czas treningu i strojenia | długi | krótki |

- Na danych **tabelarycznych** gradient boosting zwykle dorównuje sieciom lub je przewyższa (Grinsztajn i in., 2022).
- Gdy potrzebna jest interpretowalność, niski koszt, niskie opóźnienia lub dane są nieliczne, wybierz klasyczne ML.

#### Praktyczna reguła
Zacznij od prostego baseline'u (regresja logistyczna, boosting). Przejdź do DL dla danych niestrukturalnych lub gdy baseline jest niewystarczający, a masz dane i zasoby.

**Źródła:**
- [LeCun, Bengio, Hinton - Deep learning (Nature)](https://www.nature.com/articles/nature14539)
- [Bengio, Courville, Vincent - Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538)
- [Grinsztajn et al. - Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)
- [Kaplan et al. - Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)

---

<a id="q81"></a>
### 81. Czym są funkcje aktywacji i dlaczego się je stosuje?

**Odpowiedź:**

**Funkcja aktywacji** `φ` to nieliniowa funkcja stosowana element-wise do wyjścia neuronu (`a = φ(z)`, `z = Wx + b`).

#### Dlaczego są potrzebne
1. **Nieliniowość**: bez niej dowolnie głęboka sieć redukuje się do jednego przekształcenia liniowego (`W_2(W_1 x) = (W_2 W_1) x`). Nieliniowość pozwala aproksymować złożone funkcje (twierdzenie o uniwersalnej aproksymacji).
2. **Kształtowanie wyjścia**: sigmoid ogranicza do (0,1) (prawdopodobieństwo binarne), softmax daje rozkład prawdopodobieństwa, tanh do (-1,1).
3. **Wpływ na przepływ gradientu**: kształt `φ'` decyduje, czy gradient zanika, czy eksploduje.
4. **Rzadkość aktywacji**: ReLU zeruje część neuronów, dając reprezentacje rzadkie.

#### Pożądane własności
- Nieliniowość i różniczkowalność (prawie wszędzie).
- Tania obliczeniowo, gradient niezerowy w szerokim zakresie (brak nasycenia).
- Najlepiej wyśrodkowana wokół zera i monotoniczna lub gładka.

#### Wybór w praktyce
| Miejsce | Typowy wybór |
|---|---|
| Warstwy ukryte (MLP, CNN) | ReLU, Leaky ReLU, GELU/SiLU |
| Transformery | GELU, SwiGLU |
| RNN/LSTM (bramki) | sigmoid, tanh |
| Wyjście - klasyfikacja binarna | sigmoid |
| Wyjście - wieloklasowa | softmax |
| Wyjście - regresja | brak (liniowe) |

#### Uwagi
- Często softmax/sigmoid łączy się ze stratą w jednej numerycznie stabilnej funkcji (`BCEWithLogitsLoss`, `CrossEntropyLoss` przyjmują logity).
- Dobór aktywacji powinien iść w parze z inicjalizacją (He dla ReLU, Xavier dla tanh).

```python
import torch.nn as nn
layer = nn.Sequential(nn.Linear(128, 64), nn.ReLU())
```

**Źródła:**
- [Deep Learning book - Chapter 6.3: Hidden Units](https://www.deeplearningbook.org/contents/mlp.html)
- [CS231n - Neural Networks Part 1 (Activation Functions)](https://cs231n.github.io/neural-networks-1/)
- [PyTorch - Non-linear activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [Wikipedia - Activation function](https://en.wikipedia.org/wiki/Activation_function)

---

<a id="q82"></a>
### 82. Wyjaśnij funkcje aktywacji Sigmoid, Tanh, ReLU, LeakyReLU i Softmax wraz z ich zaletami i wadami.

**Odpowiedź:**

#### Sigmoid
$$\sigma(z) = \frac{1}{1+e^{-z}}, \quad \sigma'(z)=\sigma(z)(1-\sigma(z)) \le 0.25$$
- Zakres (0,1), interpretacja probabilistyczna.
- (+) Naturalna na wyjściu w klasyfikacji binarnej, w bramkach LSTM/GRU.
- (-) **Nasycenie** (gradient ~0 dla dużych |z|), maksymalna pochodna 0,25 -> zanikanie gradientu, wyjście niewyśrodkowane, kosztowne `exp`.

#### Tanh
$$\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}} = 2\sigma(2z)-1, \quad \tanh'(z)=1-\tanh^2(z)$$
- Zakres (-1,1), wyśrodkowany wokół zera.
- (+) Lepszy niż sigmoid w warstwach ukrytych (gradienty nie mają stałego znaku).
- (-) Wciąż się nasyca i ma zanikający gradient.

#### ReLU
$$\text{ReLU}(z)=\max(0,z), \quad \text{ReLU}'(z)=\mathbb{1}[z>0]$$
- (+) Tania, brak nasycenia dla z>0 (gradient 1), rzadkie aktywacje, szybka zbieżność - domyślny wybór w CNN/MLP.
- (-) **Dying ReLU**: neuron z z<0 dla wszystkich danych ma gradient 0 i „umiera”; wyjście nieujemne (niewyśrodkowane); nieróżniczkowalna w 0.

#### Leaky ReLU
$$\text{LeakyReLU}(z)=\begin{cases} z & z>0\\ \alpha z & z\le 0\end{cases}, \quad \alpha\approx 0.01$$
- (+) Niezerowy gradient dla z<0, ogranicza problem umierających neuronów. Wariant: **PReLU** (uczone `α`).
- (-) Dodatkowy hiperparametr, zysk nie zawsze wyraźny.

#### Softmax
$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
- Zamienia wektor logitów w rozkład prawdopodobieństwa (suma 1). Stosowana na wyjściu w klasyfikacji wieloklasowej i w mechanizmie attention.
- (+) Interpretowalność, różniczkowalność, w parze z cross-entropy daje gradient `ŷ - y`.
- (-) Wzajemnie wykluczające się klasy (dla multi-label użyj sigmoid), skłonność do nadmiernej pewności, wymaga stabilizacji numerycznej (odjęcie `max(z)`); `exp` może przepełnić.
- Temperatura `T` zmienia „ostrość”: `softmax(z/T)`.

#### Porównanie
| | Zakres | Wyśrodkowana | Zanik gradientu | Typowe użycie |
|---|---|---|---|---|
| Sigmoid | (0,1) | nie | tak | wyjście binarne, bramki |
| Tanh | (-1,1) | tak | tak | RNN, bramki |
| ReLU | [0,∞) | nie | nie (z>0) | warstwy ukryte |
| LeakyReLU | (-∞,∞) | ~ | nie | ukryte, gdy ReLU umiera |
| Softmax | (0,1), suma 1 | - | - | wyjście wieloklasowe |

Współcześnie w Transformerach dominują GELU i SwiGLU.

**Źródła:**
- [CS231n - Neural Networks Part 1 (Commonly used activation functions)](https://cs231n.github.io/neural-networks-1/)
- [He et al. - Delving Deep into Rectifiers (PReLU, He init)](https://arxiv.org/abs/1502.01852)
- [Hendrycks, Gimpel - Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)
- [Wikipedia - Softmax function](https://en.wikipedia.org/wiki/Softmax_function)

---

<a id="q83"></a>
### 83. Dlaczego Sigmoid i Tanh nie są preferowane w warstwach ukrytych sieci neuronowej?

**Odpowiedź:**

Głównie z powodu **zanikającego gradientu** (vanishing gradient) i związanych z nim problemów optymalizacji.

#### 1. Nasycenie i zanik gradientu
Dla dużych |z| sigmoid i tanh są płaskie, więc pochodna jest bliska 0. Maksymalna pochodna sigmoida to 0,25 (w z=0), tanh - 1 (tylko w 0). W backpropagation gradient do warstwy `l` zawiera iloczyn pochodnych aktywacji wszystkich wyższych warstw:

$$\frac{\partial L}{\partial z^{(l)}} \propto \prod_{k>l} W^{(k)\top}\,\varphi'(z^{(k)})$$

Dla sigmoida każdy czynnik ≤ 0,25, więc po 10 warstwach gradient maleje rzędu `0.25^{10} ≈ 10^{-6}` (przy wagach o rozsądnej skali). Wcześniejsze warstwy uczą się bardzo wolno lub wcale. Neurony w nasyceniu prawie nie reagują na zmiany wag.

#### 2. Wyjście sigmoida nie jest wyśrodkowane
Wyjścia w (0,1) są zawsze dodatnie, więc gradienty względem wag następnej warstwy mają ten sam znak dla wszystkich wag danego neuronu (`δ · a`, `a>0`). Powoduje to zygzakowate aktualizacje i wolniejszą zbieżność. Tanh jest wyśrodkowany wokół zera, więc ten problem jest mniejszy, i dlatego jest zwykle lepszy niż sigmoid.

#### 3. Koszt obliczeniowy
`exp` jest droższe niż `max(0,z)` (ReLU), choć na nowoczesnym sprzęcie różnica jest mniejsza.

#### Dlaczego ReLU lepiej
ReLU ma gradient 1 dla z>0 (brak nasycenia po stronie dodatniej), jest tania i daje rzadkie aktywacje, co przyspiesza zbieżność. Glorot i Bengio (2010) pokazali też, że sigmoid w głębokich sieciach z „naiwną” inicjalizacją nasyca się i spowalnia uczenie.

#### Kiedy sigmoid/tanh mają sens
- **Wyjście** klasyfikacji binarnej (sigmoid) - nie zanika, bo strata (cross-entropy) kompensuje nasycenie.
- **Bramki** w LSTM/GRU (sigmoid 0-1 jako „bramka”, tanh do kandydatów) - potrzebny ograniczony zakres.
- Płytkie sieci lub odpowiednia inicjalizacja (Xavier) i normalizacja (BatchNorm) łagodzą problem.

Alternatywy: ReLU, Leaky ReLU, GELU, SiLU, połączenia rezydualne.

**Źródła:**
- [Glorot, Bengio - Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)
- [LeCun et al. - Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
- [CS231n - Neural Networks Part 1 (Activation Functions)](https://cs231n.github.io/neural-networks-1/)

---

<a id="q84"></a>
### 84. Czym jest dropout i dlaczego jest skuteczny?

**Odpowiedź:**

**Dropout** (Srivastava, Hinton i in., 2014) to technika regularyzacji: podczas treningu każdy neuron (jego wyjście) jest z prawdopodobieństwem `p` **zerowany**, niezależnie w każdym kroku i dla każdego przykładu. W inferencji dropout jest wyłączony.

#### Mechanizm
Dla aktywacji `h` losujemy maskę `m ~ Bernoulli(1-p)`:

$$\tilde h = m \odot h$$

W popularnej wersji **inverted dropout** dzielimy dodatkowo przez `(1-p)` już podczas treningu, aby wartość oczekiwana aktywacji pozostała niezmieniona i w inferencji nie trzeba było skalować:

$$\tilde h = \frac{m \odot h}{1-p}$$

#### Dlaczego działa
1. **Zapobiega koadaptacji neuronów**: neuron nie może polegać na obecności konkretnych innych neuronów, więc musi uczyć się cech użytecznych w wielu kontekstach.
2. **Trening zespołu (ensemble)**: każdy krok trenuje inną podsieć (z `2^n` możliwych). Wagi są współdzielone, a inferencja bez dropoutu przybliża uśrednienie predykcji wykładniczo wielu podsieci (geometrycznie).
3. **Szum jako regularyzacja**: wstrzykiwanie szumu utrudnia zapamiętywanie danych treningowych, redukując overfitting. Dla modeli liniowych dropout jest zbliżony do adaptacyjnej regularyzacji L2.
4. **Bardziej odporne, rozproszone reprezentacje.**

#### Praktyka
- Typowe `p`: 0.1-0.5 (0.5 dla warstw w pełni połączonych, mniejsze dla warstw wejściowych, ~0.1 w Transformerach). W CNN częściej stosuje się BatchNorm, augmentację, `Dropout2d`/DropBlock.
- Wyższe `p` -> silniejsza regularyzacja, ryzyko underfittingu, dłuższy trening.
- Nie stosować dropoutu, gdy model jest już niedopasowany (underfit).
- Z BatchNorm bywa ostrożnie łączony (efekt wariancji, „variance shift”).
- Pamiętaj o `model.train()` / `model.eval()`.
- Warianty: DropConnect, spatial dropout, DropPath (stochastic depth), Monte Carlo dropout (dropout w inferencji do szacowania niepewności, Gal i Ghahramani).

```python
model = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(128, 10))
model.train()  # dropout aktywny
model.eval()   # dropout wyłączony
```

**Źródła:**
- [Outcome School - Dropout in Neural Networks](https://outcomeschool.com/blog/dropout-in-neural-networks)
- [Srivastava et al. - Dropout: A Simple Way to Prevent Neural Networks from Overfitting (JMLR)](https://jmlr.org/papers/v15/srivastava14a.html)
- [Hinton et al. - Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)
- [PyTorch - torch.nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)

---

<a id="q85"></a>
### 85. Jaki jest wpływ dropoutu na szybkość treningu i inferencji?

**Odpowiedź:**

#### Trening
- **Koszt obliczeniowy pojedynczego kroku**: minimalnie wyższy. Generowanie maski i mnożenie element-wise to tanie operacje, ale dropout **nie zmniejsza** liczby FLOPów (w typowych implementacjach neurony są zerowane maską, a mnożenia macierzowe wykonywane w pełnym rozmiarze). Dodatkowo przechowywana jest maska dla backward pass.
- **Liczba kroków do zbieżności**: zwykle **większa**. Szum w aktualizacjach spowalnia zbieżność, a efektywna pojemność jest mniejsza; często potrzeba więcej epok (rząd wielkości: kilkadziesiąt procent więcej, zależnie od zadania i `p`). W zamian zmniejsza się luka generalizacji.
- Wysoki dropout może powodować underfitting i wolniejszy spadek treningowej straty. Funkcja straty w treningu jest wyższa i bardziej zaszumiona niż w ewaluacji (bo trening ma losowe maskowanie).
- W dużych LLM z jednym przejściem przez dane (epoka < 1) dropout często się pomija, ponieważ overfitting nie jest problemem.

#### Inferencja
- Dropout jest **wyłączony** (`model.eval()`), więc nie ma dodatkowego kosztu: to zwykły forward pass całej sieci.
- Skalowanie: w inverted dropout skalowanie wykonywane jest w treningu (`/(1-p)`), więc w inferencji nic się nie zmienia. W wersji oryginalnej wagi mnoży się przez `(1-p)` (jednorazowo).
- **Wyjątek**: Monte Carlo dropout - dropout zostaje włączony i wykonuje się `T` przejść forward (np. 20-50), by oszacować niepewność, co daje `T`-krotnie wolniejszą inferencję.

#### Podsumowanie
| Faza | Wpływ |
|---|---|
| Krok treningowy | prawie brak (lekki narzut na maskę) |
| Całkowity czas treningu | zwykle dłuższy (więcej epok) |
| Inferencja | brak wpływu (dropout wyłączony) |
| MC dropout | T razy wolniej |

#### Uwaga praktyczna
Jeśli zapomnisz o `model.eval()`, predykcje będą losowe i gorsze. Dla przyspieszenia treningu dropout może być zastąpiony innymi regularyzatorami (augmentacja, weight decay, early stopping).

**Źródła:**
- [Outcome School - Dropout in Neural Networks](https://outcomeschool.com/blog/dropout-in-neural-networks)
- [Srivastava et al. - Dropout (JMLR)](https://jmlr.org/papers/v15/srivastava14a.html)
- [Gal, Ghahramani - Dropout as a Bayesian Approximation](https://arxiv.org/abs/1506.02142)

---

<a id="q86"></a>
### 86. Czym jest regularyzacja L1/L2 i jak wpływa na sieć neuronową?

**Odpowiedź:**

**Regularyzacja** dodaje do funkcji straty karę za złożoność modelu (wielkość wag), redukując overfitting.

$$L_{total} = L_{data}(\theta) + \lambda\,\Omega(\theta)$$

#### L2 (weight decay, ridge)
$$\Omega(\theta) = \tfrac12\sum_i w_i^2 \quad\Rightarrow\quad \nabla_w = \nabla L_{data} + \lambda w$$

Aktualizacja SGD: $w \leftarrow (1-\eta\lambda)\,w - \eta\nabla L_{data}$ - wagi są w każdym kroku „kurczone” (**weight decay**).
- Preferuje **małe, rozproszone** wagi, nie zeruje ich (rozwiązanie gęste).
- Gładszy model, mniejsza wrażliwość na szum w wejściu.
- Interpretacja bayesowska: prior gaussowski na wagach (MAP).

#### L1 (lasso)
$$\Omega(\theta) = \sum_i |w_i| \quad\Rightarrow\quad \nabla_w = \nabla L_{data} + \lambda\,\text{sign}(w)$$

- Stały „ciąg” ku zeru niezależnie od wielkości wagi, więc prowadzi do **rzadkich** wag (wiele dokładnie ~0), czyli selekcji cech i kompresji.
- Interpretacja bayesowska: prior Laplace'a.
- Nieróżniczkowalna w 0 (subgradient, w praktyce wagi oscylują wokół zera; stosuje się proximal methods).

#### Intuicja geometryczna
Zbiory poziomicowe kary L1 to romby z ostrymi rogami na osiach, więc optimum łatwo trafia w róg (zerowa współrzędna). Dla L2 kula jest gładka, więc zerowanie jest nietypowe.

#### Wpływ na sieć neuronową
- Zmniejsza overfitting i poprawia generalizację (wymienia wariancję na niewielkie obciążenie).
- Zapobiega niekontrolowanemu wzrostowi wag i stabilizuje trening.
- L1 daje rzadkie sieci (użyteczne do pruningu), L2 - standard w praktyce.
- **Elastic Net**: `λ1‖w‖1 + λ2‖w‖2²`.
- Zwykle nie regularyzuje się biasów ani parametrów normalizacji.

#### Praktyka
- `λ` (weight decay) zwykle 1e-5 - 1e-2, strojone na walidacji.
- W Adamie klasyczne L2 w gradiencie różni się od prawdziwego weight decay, dlatego zaleca się **AdamW** (decoupled decay).
- Przy BatchNorm efekt L2 na wagi przed normalizacją to głównie zmiana efektywnego learning rate, a nie klasyczna regularyzacja.

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)   # L2 (decoupled)
l1 = 1e-5 * sum(p.abs().sum() for p in model.parameters())
loss = criterion(out, y) + l1                                              # L1
```

**Źródła:**
- [Outcome School - Regularization In Machine Learning](https://outcomeschool.com/blog/regularization-in-machine-learning)
- [Deep Learning book - Chapter 7: Regularization for Deep Learning](https://www.deeplearningbook.org/contents/regularization.html)
- [Loshchilov, Hutter - Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [scikit-learn - Linear Models (Ridge, Lasso, Elastic-Net)](https://scikit-learn.org/stable/modules/linear_model.html)

---

<a id="q87"></a>
### 87. Czym jest normalizacja wsadowa (batch normalization) i dlaczego się ją stosuje?

**Odpowiedź:**

**Batch Normalization (BN)**, Ioffe i Szegedy (2015), normalizuje aktywacje warstwy statystykami z bieżącego mini-batcha, a następnie skaluje i przesuwa je uczonymi parametrami.

#### Algorytm
Dla cechy `x` w mini-batchu o rozmiarze `m`:

$$\mu_B = \frac1m\sum_i x_i,\qquad \sigma_B^2 = \frac1m\sum_i (x_i-\mu_B)^2$$
$$\hat x_i = \frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y_i = \gamma\,\hat x_i + \beta$$

- `γ`, `β` - parametry uczone (pozwalają sieci odtworzyć tożsamość, jeśli normalizacja nie jest korzystna).
- W CNN statystyki liczone są per kanał po wymiarach batch, H, W.
- W **inferencji** używa się **średnich kroczących** (running mean/var) zebranych w treningu, a nie statystyk batcha.

#### Dlaczego się stosuje
1. **Stabilniejszy i szybszy trening**: pozwala na wyższy learning rate.
2. **Mniejsza wrażliwość na inicjalizację.**
3. **Łagodzi zanikanie/eksplozję gradientów** i poprawia kondycję krajobrazu funkcji straty (Santurkar i in. 2018: wygładzenie optymalizacji, a niekoniecznie redukcja „internal covariate shift”, jak pierwotnie tłumaczono).
4. **Lekka regularyzacja**: szum ze statystyk batcha działa podobnie do dropoutu, więc często można zmniejszyć dropout.

#### Wady
- Zależność od rozmiaru batcha: przy małych batchach (np. <8-16) statystyki są zaszumione.
- Różne zachowanie train vs eval (błędy przy zapomnianym `eval()`), problemy w RNN i w treningu rozproszonym (synchronizacja statystyk: SyncBatchNorm).
- Nieodpowiednie dla sekwencji o zmiennej długości: tam stosuje się **LayerNorm** (Transformery), a także GroupNorm, InstanceNorm, RMSNorm.

#### Praktyka
- Kolejność: `Conv/Linear -> BN -> aktywacja` (najczęściej), bias w warstwie poprzedzającej jest zbędny.
- Domyślnie w CNN (ResNet); w Transformerach LayerNorm/RMSNorm.

```python
block = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1, bias=False),
                      nn.BatchNorm2d(64), nn.ReLU())
```

**Źródła:**
- [Outcome School - Batch Normalization vs Layer Normalization](https://outcomeschool.com/blog/batch-normalization-vs-layer-normalization)
- [Ioffe, Szegedy - Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167)
- [Santurkar et al. - How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604)
- [PyTorch - torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)

---

<a id="q88"></a>
### 88. Jakie hiperparametry normalizacji wsadowej (batch normalization) można optymalizować?

**Odpowiedź:**

Sama warstwa BN ma niewiele hiperparametrów, ale kilka decyzji wokół niej wpływa na wyniki. Trzeba odróżnić **parametry uczone** (`γ`, `β`) od **hiperparametrów**.

#### Parametry uczone (nie hiperparametry)
- `γ` (scale) i `β` (shift) - trenowane gradientem, inicjalizowane zwykle `γ=1`, `β=0`. Ich inicjalizacja jest strojona rzadko (np. `γ=0` w ostatniej BN bloku rezydualnego - „zero-init residual”, Goyal i in., stabilizuje trening dużych batchy).
- `running_mean`, `running_var` - statystyki wyznaczane, nie uczone gradientem.

#### Hiperparametry
1. **`momentum`** - współczynnik aktualizacji średnich kroczących: `running = (1 - momentum)·running + momentum·batch_stat`. W PyTorch domyślnie 0.1 (w Keras/TF konwencja odwrotna, 0.99). Zbyt duży -> zaszumione statystyki inferencji; zbyt mały -> opóźniona adaptacja (istotne przy krótkim treningu lub fine-tuningu na innej domenie).
2. **`epsilon (eps)`** - stała numeryczna (1e-5 domyślnie w PyTorch, 1e-3 w Keras). Chroni przed dzieleniem przez 0; większe wartości bywają stabilniejsze przy niskiej precyzji (fp16).
3. **`affine`** (`center`/`scale` w Keras) - czy uczyć `γ` i `β`. Wyłączenie zmniejsza pojemność.
4. **`track_running_stats`** - czy śledzić statystyki. Gdy `False`, używa się statystyk batcha także w inferencji.
5. **Rozmiar batcha** - najważniejszy „pośredni” hiperparametr; wpływa na jakość statystyk. Przy małych batchach rozważ GroupNorm, LayerNorm lub Ghost/Sync BN.
6. **Pozycja BN**: przed czy po aktywacji (`conv-BN-ReLU` standard, ale strojone).
7. **Wymiary normalizacji**: BN1d/2d/3d, ewentualnie GroupNorm z liczbą grup.
8. **Weight decay dla `γ`, `β`** - często ustawia się zero.
9. **Zamrożenie BN** (tryb `eval` dla warstw BN) przy fine-tuningu z małym batchem.

#### Jak strojić
- Zacznij od wartości domyślnych, zmieniaj `momentum` tylko przy problemach z rozbieżnością między metrykami train/eval.
- Porównaj warianty normalizacji (BN vs GN vs LN) na walidacji.
- Wraz z BN można zwiększyć learning rate i zmniejszyć dropout.

```python
bn = nn.BatchNorm1d(128, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
```

**Źródła:**
- [PyTorch - torch.nn.BatchNorm1d](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
- [Ioffe, Szegedy - Batch Normalization](https://arxiv.org/abs/1502.03167)
- [Goyal et al. - Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
- [Wu, He - Group Normalization](https://arxiv.org/abs/1803.08494)

---

<a id="q89"></a>
### 89. Czym jest współdzielenie parametrów (parameter sharing) w deep learningu?

**Odpowiedź:**

**Parameter sharing** oznacza, że ten sam zestaw wag jest używany w wielu miejscach modelu (w różnych pozycjach wejścia, krokach czasu lub warstwach), zamiast uczyć osobne parametry dla każdego miejsca.

#### Przykłady
1. **CNN**: jeden filtr (kernel) przesuwany po całym obrazie. Ta sama detekcja (np. krawędzi) działa w każdym miejscu. Efekt: ekwiwariancja względem translacji i radykalnie mniej parametrów. Warstwa gęsta dla obrazu 224×224×3 -> 1000 neuronów to ~150 mln wag, a splot 3×3 z 64 filtrami tylko `3·3·3·64 + 64 = 1792`.
2. **RNN/LSTM/GRU**: te same wagi przy każdym kroku czasu. Model obsługuje sekwencje o dowolnej długości i uogólnia wzorce niezależnie od pozycji.
3. **Transformer**: te same wagi (attention i FFN) stosowane do każdego tokenu; ponadto **weight tying** (wspólna macierz embeddingów wejściowych i wyjściowych w modelach językowych) oraz cross-layer sharing (ALBERT, Universal Transformer).
4. **Sieci syjamskie**: identyczne enkodery dla dwóch wejść (podobieństwo, face verification).
5. **Multi-task learning**: wspólny trzon (backbone) z osobnymi głowami. Grouped/multi-query attention dzielą klucze i wartości między głowami.
6. **Autoenkodery**: „tied weights” (`W_dec = W_encᵀ`).

#### Zalety
- **Mniej parametrów** -> mniejsze zużycie pamięci, niższe ryzyko overfittingu.
- **Wbudowane założenia (inductive bias)**: lokalność, niezależność od pozycji/czasu, symetria.
- **Lepsza efektywność danych** i generalizacja.
- Większa liczba gradientów spływających do tych samych wag (uczenie statystycznie mocniejsze).

#### Wady i uwagi
- Mniejsza elastyczność, gdy założenie jest błędne (np. obrazy wyrównane, gdzie pozycja ma znaczenie: twarze - stosuje się lokalnie łączone warstwy).
- Gradienty dla współdzielonych wag są **sumowane** po wszystkich miejscach użycia (w RNN: BPTT).
- Współdzielenie między warstwami zmniejsza pojemność przy tym samym koszcie obliczeń.

```python
model.lm_head.weight = model.embed.weight   # weight tying
```

**Źródła:**
- [Deep Learning book - Chapter 9: Convolutional Networks (Parameter sharing)](https://www.deeplearningbook.org/contents/convnets.html)
- [Press, Wolf - Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859)
- [Lan et al. - ALBERT: A Lite BERT](https://arxiv.org/abs/1909.11942)
- [CS231n - Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)

---

<a id="q90"></a>
### 90. Czym jest uczenie reprezentacji (representation learning) i dlaczego jest użyteczne?

**Odpowiedź:**

**Representation learning** to uczenie się przez model użytecznych **reprezentacji** (cech, embeddingów) surowych danych, zamiast ręcznego projektowania cech (feature engineering). Reprezentacja `z = f(x)` powinna zachowywać informacje istotne dla zadań, a odrzucać szum i zbędne detale.

#### Jak to działa
- W sieci głębokiej kolejne warstwy tworzą **hierarchię**: w CNN krawędzie -> tekstury -> części -> obiekty; w NLP litery/tokeny -> składnia -> semantyka.
- Reprezentacja końcowej warstwy ukrytej jest niemal liniowo separowalna dla zadania (dlatego „linear probe” to standardowy test jakości cech).

#### Sposoby uczenia
- **Nadzorowane**: aktywacje sieci wytrenowanej na ImageNet.
- **Nienadzorowane**: autoenkodery, word2vec, PCA (liniowo), VAE.
- **Samonadzorowane (self-supervised)**: masked language modeling (BERT), przewidywanie następnego tokenu (GPT), kontrastowe (SimCLR, CLIP), masked autoencoders. Etykiety generowane z danych.
- **Multi-task** i **transfer learning**.

#### Dlaczego jest użyteczne
1. **Eliminuje ręczną inżynierię cech.**
2. **Transfer**: reprezentacje z dużych, nieoznakowanych zbiorów przenoszą się na zadania z małą liczbą etykiet (fine-tuning, few-shot).
3. **Redukcja wymiarowości** i kompaktowość: embedding 768-D zamiast tysięcy rzadkich cech.
4. **Uchwycenie podobieństwa**: bliskie punkty w przestrzeni embeddingów to semantycznie bliskie obiekty (wyszukiwanie wektorowe, RAG, rekomendacje).
5. **Rozplątanie (disentanglement)** czynników zmienności i odporność na niepożądane transformacje.
6. **Multimodalność**: wspólna przestrzeń dla tekstu i obrazu (CLIP).

#### Własności dobrej reprezentacji (Bengio i in.)
Gładkość, hierarchia, wspólne czynniki dla zadań, rzadkość, rozplątanie, naturalne skupiska.

#### Pułapki
- Reprezentacje mogą kodować uprzedzenia z danych.
- Słaba interpretowalność.
- Dryf dziedziny: reprezentacje wytrenowane w jednej domenie mogą źle działać w innej.
- Ocena: linear probing, kNN, wydajność na zadaniach docelowych.

**Źródła:**
- [Bengio, Courville, Vincent - Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538)
- [Deep Learning book - Chapter 15: Representation Learning](https://www.deeplearningbook.org/contents/representation.html)
- [Chen et al. - A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709)
- [Radford et al. - Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)

---

<a id="q91"></a>
### 91. Czym jest model generatywny i czym różni się od modelu dyskryminatywnego?

**Odpowiedź:**

#### Definicje
- **Model dyskryminatywny** uczy się bezpośrednio **`p(y | x)`** (lub granicy decyzyjnej `x -> y`): „jaka etykieta dla tych danych?”. Przykłady: regresja logistyczna, SVM, drzewa decyzyjne, lasy losowe, boosting, standardowe sieci klasyfikujące.
- **Model generatywny** uczy się rozkładu wspólnego **`p(x, y)`** (dla nienadzorowanych: `p(x)`), czyli tego, jak dane są generowane. Umożliwia **próbkowanie** nowych danych. Przykłady: Naive Bayes, GMM, HMM, VAE, GAN, modele autoregresyjne (GPT), modele dyfuzyjne, flow-based.

Klasyfikację generatywną wykonuje się przez regułę Bayesa:

$$p(y\mid x) = \frac{p(x\mid y)\,p(y)}{p(x)}$$

#### Porównanie
| Cecha | Dyskryminatywny | Generatywny |
|---|---|---|
| Uczy się | `p(y|x)` | `p(x,y)` lub `p(x)` |
| Cel | najlepsza predykcja | modelowanie danych |
| Generowanie próbek | nie | tak |
| Ilość danych | zwykle mniej dla dobrej klasyfikacji | zwykle więcej |
| Dokładność klasyfikacji | zazwyczaj wyższa asymptotycznie | zwykle niższa przy silnych założeniach |
| Brakujące dane | trudniej | naturalnie (marginalizacja) |
| Detekcja anomalii | pośrednio | naturalnie (niskie `p(x)`) |
| Złożoność | niższa | wyższa |

#### Kompromisy
- Ng i Jordan (2001): generatywny Naive Bayes osiąga swój (gorszy) błąd asymptotyczny szybciej (przy mniejszej liczbie próbek) niż regresja logistyczna, ale przy dużej ilości danych wygrywa dyskryminatywna.
- Dyskryminatywne skupiają zdolności na granicy decyzji, generatywne muszą modelować całą strukturę danych (w tym niepotrzebne detale).

#### Zastosowania modeli generatywnych
Generowanie tekstu/obrazów/dźwięku/kodu, augmentacja danych, uzupełnianie brakujących danych (inpainting), wykrywanie anomalii, kompresja, uczenie półnadzorowane, symulacja.

#### Uwagi
Współczesne LLM to modele generatywne (autoregresyjne `p(x_t | x_<t)`), które po instrukcjach potrafią działać jak dyskryminatywne (zero-shot klasyfikacja). Granica jest więc rozmyta. Dyskryminator w GAN jest modelem dyskryminatywnym w grze z generatywnym.

**Źródła:**
- [Wikipedia - Generative model](https://en.wikipedia.org/wiki/Generative_model)
- [Wikipedia - Discriminative model](https://en.wikipedia.org/wiki/Discriminative_model)
- [Goodfellow - NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)
- [scikit-learn - Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)

---

<a id="q92"></a>
### 92. Wyjaśnij, jak działa model generatywny.

**Odpowiedź:**

Cel modelu generatywnego: nauczyć się rozkładu danych `p_data(x)`, aby móc **próbkować** nowe przykłady `x ~ p_θ(x)` przypominające dane treningowe. Realizuje się to na kilka sposobów.

#### Ogólny schemat
1. Zbierz dane treningowe `x ~ p_data`.
2. Zdefiniuj rodzinę modeli `p_θ(x)` (sieć neuronowa).
3. Dobierz `θ`, minimalizując odległość (np. KL) między `p_data` a `p_θ`, co w praktyce oznacza maksymalizację log-wiarygodności (lub jej dolnego ograniczenia).
4. Generuj: losuj szum/kontekst i przepuszczaj przez wytrenowany model.

#### Główne rodziny
| Rodzina | Idea | Trening | Przykłady |
|---|---|---|---|
| **Autoregresyjne** | $p(x)=\prod_t p(x_t\mid x_{<t})$ | dokładna wiarygodność | GPT, PixelCNN |
| **VAE** | enkoder + dekoder z latentem `z` | ELBO | VAE, VQ-VAE |
| **GAN** | generator vs dyskryminator | gra minimax | StyleGAN |
| **Flow-based** | odwracalne transformacje | dokładna wiarygodność | RealNVP, Glow |
| **Dyfuzyjne** | uczenie odszumiania kolejnych kroków szumu | uproszczony wariant ELBO | DDPM, Stable Diffusion |
| **Energy-based** | funkcja energii | contrastive divergence | EBM |

#### Przykład: model autoregresyjny (LLM)
Trening: maksymalizacja $\sum_t \log p_\theta(x_t\mid x_{<t})$ (cross-entropy). Generowanie: losujemy token z rozkładu wyjścia (temperatura, top-k, top-p), dołączamy do kontekstu i powtarzamy.

#### Przykład: VAE/GAN
Losujemy `z ~ N(0, I)` i przepuszczamy przez dekoder/generator `G(z)`, który mapuje prosty rozkład na złożony rozkład danych.

#### Przykład: dyfuzja
Proces „w przód” dodaje szum gaussowski do obrazu w `T` krokach. Sieć uczy się przewidywać szum (lub oryginał), a generowanie startuje od czystego szumu i iteracyjnie go usuwa.

#### Wyzwania
- Ocena jakości (FID, Inception Score, perplexity, oceny ludzkie).
- Kompromis: jakość próbek, różnorodność, szybkość generacji (dyfuzja - wiele kroków; GAN - jeden krok, ale niestabilny trening), tractability wiarygodności.
- Halucynacje, uprzedzenia, prawa autorskie i nadużycia.

**Źródła:**
- [Goodfellow - NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)
- [Kingma, Welling - Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Ho et al. - Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Stanford CS236 - Deep Generative Models (notatki)](https://deepgenerativemodels.github.io/notes/)

---

<a id="q93"></a>
### 93. Wyjaśnij architekturę koder-dekoder (Encoder-Decoder).

**Odpowiedź:**

**Encoder-Decoder** to architektura złożona z dwóch części:
- **Enkoder** przekształca wejście `x` w zwartą reprezentację (kontekst, wektor/tensor ukryty `z` lub sekwencję reprezentacji),
- **Dekoder** generuje wyjście `y` na podstawie tej reprezentacji.

$$z = \text{Enc}(x), \qquad y = \text{Dec}(z)$$

#### Historia i warianty
1. **Seq2seq z RNN/LSTM** (Sutskever i in., Cho i in., 2014): enkoder kompresuje zdanie do jednego wektora, dekoder generuje tłumaczenie token po tokenie. Wąskim gardłem był stały rozmiar wektora dla długich zdań.
2. **Attention** (Bahdanau i in., 2014): dekoder w każdym kroku „zagląda” do wszystkich stanów enkodera i ważonym uśrednieniem wybiera istotne fragmenty wejścia.
3. **Transformer** (Vaswani i in., 2017): enkoder ma bloki self-attention + FFN; dekoder ma **masked self-attention** (patrzy tylko w przeszłość), **cross-attention** (zapytania z dekodera, klucze i wartości z enkodera) i FFN. Dekoder działa autoregresyjnie:

$$p(y\mid x)=\prod_t p(y_t\mid y_{<t}, \text{Enc}(x))$$

4. **Konwolucyjne**: U-Net (segmentacja, dyfuzja) - enkoder zmniejsza rozdzielczość, dekoder ją odbudowuje, połączenia skip przenoszą szczegóły.
5. **Autoenkodery i VAE**: enkoder -> latent -> dekoder rekonstruuje wejście.

#### Zastosowania
- Tłumaczenie maszynowe, streszczanie (T5, BART), rozpoznawanie mowy (Whisper), opis obrazów (CNN + dekoder tekstowy), OCR, segmentacja (U-Net), generacja obrazów (latent diffusion), kompresja.

#### Trening
- **Teacher forcing**: podczas treningu dekoder dostaje poprawne poprzednie tokeny, w inferencji własne predykcje (exposure bias). Strata: cross-entropy po tokenach.
- Dekodowanie: greedy, beam search, sampling.

#### Zalety i wady
- (+) Elastyczne wejście/wyjście o różnych długościach i modalnościach; naturalne dla zadań sekwencja-do-sekwencji.
- (-) Koszt dwóch komponentów; w wersji RNN wąskie gardło i sekwencyjność; autoregresyjna generacja jest wolna.

Modele tylko-dekoderowe (GPT) i tylko-enkoderowe (BERT) to uproszczone warianty.

**Źródła:**
- [Outcome School - Decoding Transformer Architecture](https://outcomeschool.com/blog/decoding-transformer-architecture)
- [Sutskever et al. - Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Bahdanau et al. - Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Vaswani et al. - Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

<a id="q94"></a>
### 94. Jaka jest różnica między architekturami Transformer typu encoder-only, decoder-only i encoder-decoder?

**Odpowiedź:**

Wszystkie bazują na blokach z attention i FFN, ale różnią się **maską uwagi** i sposobem użycia.

#### Encoder-only (np. BERT, RoBERTa, DeBERTa)
- **Dwukierunkowe** self-attention: każdy token widzi cały kontekst (lewy i prawy).
- Trening: **masked language modeling** (przewidywanie zamaskowanych tokenów).
- Wyjście: kontekstowe reprezentacje tokenów (embeddingi).
- Zadania: klasyfikacja, NER, wyszukiwanie/embeddingi zdań, ranking, QA ekstrakcyjne (rozumienie tekstu).
- Nie generują tekstu w sposób naturalny.

#### Decoder-only (np. GPT, Llama, Mistral, Claude)
- **Przyczynowe (causal)** self-attention: token widzi tylko poprzednie tokeny (maska trójkątna).
- Trening: **next-token prediction**: $\max \sum_t \log p(x_t\mid x_{<t})$.
- Generacja autoregresyjna z KV-cache.
- Zadania: generacja tekstu, czat, kod, rozumowanie; poprzez prompt (in-context learning) także klasyfikacja i tłumaczenie.
- Obecnie dominująca architektura LLM: prosty cel treningu, dobre skalowanie i jedna architektura dla wszystkich zadań.

#### Encoder-decoder (np. T5, BART, mT5, Whisper, oryginalny Transformer)
- Enkoder (dwukierunkowy) koduje wejście; dekoder (przyczynowy) generuje wyjście, korzystając z **cross-attention** do enkodera.
- Trening: denoising/span corruption (T5, BART) lub supervised seq2seq.
- Zadania: tłumaczenie, streszczanie, transkrypcja mowy, generacja warunkowana, gdy wejście i wyjście są wyraźnie rozdzielone.

#### Porównanie
| Cecha | Encoder-only | Decoder-only | Encoder-decoder |
|---|---|---|---|
| Uwaga | dwukierunkowa | przyczynowa | dwukier. + przycz. + cross |
| Cel treningu | MLM | next-token | seq2seq / span corruption |
| Generacja | nie | tak | tak |
| Reprezentacje | bogate, kontekstowe | wyprowadzane z ukrytych stanów | oba |
| Typowy przykład | BERT | GPT/Llama | T5 |
| Koszt inferencji | jedno przejście | tokeny sekwencyjnie (KV-cache) | enkoder raz + dekoder sekwencyjnie |

#### Kiedy co wybrać
- Rozumienie i embeddingi przy niskim koszcie: encoder-only.
- Uniwersalna generacja, chat, agent: decoder-only.
- Wyraźne zadanie wejście->wyjście (tłumaczenie, ASR), mniejsze modele: encoder-decoder.

Decoder-only zwykle wygrywa w skalowaniu, ale enkodery pozostają tańsze i skuteczne w klasyfikacji i wyszukiwaniu.

**Źródła:**
- [Outcome School - Encoder vs Decoder in Transformers](https://outcomeschool.com/blog/encoder-vs-decoder-in-transformers)
- [Devlin et al. - BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Raffel et al. - Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)
- [Hugging Face - Summary of the models](https://huggingface.co/docs/transformers/model_summary)

---

<a id="q95"></a>
### 95. Czym jest przestrzeń ukryta (latent space)?

**Odpowiedź:**

**Latent space** (przestrzeń ukryta/latentna) to niskowymiarowa, abstrakcyjna przestrzeń reprezentacji, w której model koduje **czynniki zmienności** danych. Zmienne „ukryte” `z` nie są bezpośrednio obserwowane, ale model uczy się je wnioskować z danych `x`.

#### Intuicja
Obraz twarzy ma miliony pikseli, ale zmienia się w niewielu istotnych wymiarach: poza, wiek, uśmiech, oświetlenie. Przestrzeń latentna to zwięzła „mapa” takich czynników. Podobne obiekty leżą blisko siebie.

#### Gdzie występuje
- **Autoenkodery**: wąskie gardło między enkoderem a dekoderem.
- **VAE**: latent probabilistyczny `z ~ N(μ, σ²)`, regularyzowany do rozkładu `N(0, I)`.
- **GAN**: wektor szumu `z` wejściowy dla generatora (StyleGAN ma dodatkowo przestrzeń `W`).
- **Modele dyfuzyjne latentne** (Stable Diffusion): dyfuzja zachodzi w latencie kompresującym obraz, co obniża koszt.
- **Embeddingi** (word2vec, BERT, CLIP): reprezentacje w przestrzeni, gdzie odległość koduje podobieństwo semantyczne.
- Ukryte stany warstw sieci głębokich.

#### Własności i zastosowania
- **Redukcja wymiarowości** i kompresja (`x ∈ R^n -> z ∈ R^d, d ≪ n`).
- **Interpolacja**: ruch po odcinku `z = (1-t)z_1 + t z_2` daje płynne przejście (morphing) między obiektami. Dobra latentna jest ciągła i gładka.
- **Arytmetyka wektorowa**: klasyczny przykład word2vec `king - man + woman ≈ queen`; kierunki latentne odpowiadają atrybutom (edycja obrazów).
- **Generacja**: próbkowanie `z` z prior i dekodowanie.
- Wyszukiwanie podobieństwa, klasteryzacja, detekcja anomalii (duży błąd rekonstrukcji lub niski `p(z)`).
- Wizualizacja (t-SNE, UMAP) dla wglądu w to, czego model się nauczył.

#### Uwagi
- Wymiary latentne nie zawsze są interpretowalne (rozplątanie wymaga specjalnych technik: β-VAE).
- Zbyt mała przestrzeń -> utrata informacji; zbyt duża -> brak kompresji i słabsza struktura.
- W zwykłym autoenkoderze przestrzeń może mieć „dziury”, gdzie dekoder daje bezsensowne wyniki (dlatego VAE narzuca regularność).

**Źródła:**
- [Wikipedia - Latent space](https://en.wikipedia.org/wiki/Latent_space)
- [Bengio et al. - Representation Learning: A Review and New Perspectives](https://arxiv.org/abs/1206.5538)
- [Rombach et al. - High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Mikolov et al. - Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)

---

<a id="q96"></a>
### 96. Czym są autoenkodery (autoencoders)? Wyjaśnij ich warstwy i praktyczne zastosowania.

**Odpowiedź:**

**Autoenkoder** to sieć neuronowa uczona nienadzorowanie (samonadzorowanie) do **odtwarzania własnego wejścia** przez wąskie gardło, co zmusza ją do nauczenia się zwartej reprezentacji.

$$z = f_\theta(x), \quad \hat x = g_\phi(z), \quad L = \|x-\hat x\|^2 \;\text{(lub BCE)}$$

#### Warstwy
1. **Enkoder** `f`: warstwy (dense lub conv) zmniejszające wymiar, np. 784 -> 256 -> 64 -> 32.
2. **Warstwa ukryta / kod (bottleneck)** `z`: najmniejszy wymiar, zawiera skompresowaną reprezentację.
3. **Dekoder** `g`: symetryczna struktura zwiększająca wymiar (dense, transposed conv, upsampling), 32 -> 64 -> 256 -> 784.
4. Wyjście o kształcie wejścia (sigmoid dla pikseli 0-1, liniowe dla ciągłych).

Bez nieliniowości i przy stracie MSE autoenkoder liniowy uczy się tej samej podprzestrzeni co **PCA**. Dzięki nieliniowościom uczy nieliniowych rozmaitości.

#### Warianty i regularyzacja (żeby nie uczył się tożsamości)
- **Undercomplete** - `dim(z) < dim(x)`.
- **Sparse AE** - kara L1/KL na aktywacje.
- **Denoising AE** - na wejściu zaszumione `x̃`, cel: czysty `x`; uczy się odporności i struktury danych (Vincent i in.).
- **Contractive AE** - kara na normę Jacobianu enkodera.
- **Variational AE** - probabilistyczny latent, model generatywny.
- **Convolutional AE** - dla obrazów; **LSTM AE** - dla szeregów czasowych; **U-Net** - z połączeniami skip.

#### Zastosowania praktyczne
- Redukcja wymiarowości i wizualizacja.
- **Detekcja anomalii**: model trenowany na danych normalnych ma duży błąd rekonstrukcji dla anomalii (fraud, defekty przemysłowe, monitoring).
- **Odszumianie** obrazów/dźwięku, inpainting, super-rozdzielczość.
- Kompresja (stratna), pretrening reprezentacji, wyszukiwanie podobnych obiektów.
- Systemy rekomendacyjne, generacja (VAE).

#### Pułapki
- Zbyt duża pojemność -> kopiowanie wejścia (identity).
- MSE daje rozmyte rekonstrukcje obrazów.
- Jakość kompresji gorsza niż wyspecjalizowanych kodeków, a próg detekcji anomalii wymaga kalibracji.

```python
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 32))
        self.dec = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())
    def forward(self, x): return self.dec(self.enc(x))
```

**Źródła:**
- [Deep Learning book - Chapter 14: Autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html)
- [Vincent et al. - Stacked Denoising Autoencoders (JMLR)](https://www.jmlr.org/papers/v11/vincent10a.html)
- [Wikipedia - Autoencoder](https://en.wikipedia.org/wiki/Autoencoder)
- [Keras blog - Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)

---

<a id="q97"></a>
### 97. Czym jest autoenkoder wariacyjny (VAE) i czym różni się od tradycyjnego autoenkodera?

**Odpowiedź:**

**Variational Autoencoder** (Kingma i Welling, 2013) to **probabilistyczny model generatywny**. Zamiast kodować wejście do pojedynczego punktu, enkoder zwraca **rozkład** w przestrzeni latentnej, a dekoder odtwarza dane z próbki z tego rozkładu.

#### Architektura
- Enkoder (sieć wnioskowania) `q_φ(z|x)`: zwraca `μ(x)` i `log σ²(x)` rozkładu gaussowskiego `N(μ, σ²I)`.
- Próbkowanie z **reparametryzacją**: $z = \mu + \sigma\odot\epsilon,\ \epsilon\sim N(0,I)$, dzięki czemu gradient przepływa przez losowanie.
- Dekoder `p_θ(x|z)`.
- Prior: `p(z) = N(0, I)`.

#### Funkcja celu (ELBO)

$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{rekonstrukcja}} - \underbrace{D_{KL}\big(q_\phi(z|x)\,\|\,p(z)\big)}_{\text{regularyzacja}}$$

Dla gaussowskiego `q` i standardowego prioru: $D_{KL} = -\tfrac12\sum_j\left(1+\log\sigma_j^2-\mu_j^2-\sigma_j^2\right)$. Maksymalizacja ELBO to maksymalizacja dolnego ograniczenia na `log p(x)`.

#### Różnice względem klasycznego autoenkodera
| Cecha | Autoenkoder | VAE |
|---|---|---|
| Kod | punkt `z` (deterministyczny) | rozkład `q(z|x)` |
| Strata | rekonstrukcja | rekonstrukcja + KL |
| Struktura latentu | dowolna, mogą być „dziury” | ciągła, gładka, zbliżona do prioru |
| Generowanie | zwykle nie (brak prioru) | tak: `z ~ N(0,I)` -> dekoder |
| Cel | kompresja, cechy | model generatywny, wnioskowanie |
| Interpolacja | mniej wiarygodna | płynna, sensowne wyniki |

#### Zalety i wady
- (+) Stabilny trening, sensownie uporządkowany latent, możliwość próbkowania, wnioskowanie o `z`.
- (-) Rozmyte obrazy (gaussowska wiarygodność, uśrednianie), **posterior collapse** (dekoder ignoruje `z`, KL -> 0, szczególnie z silnym dekoderem autoregresyjnym).
- Warianty: **β-VAE** (waga `β` przy KL - rozplątanie), **VQ-VAE** (dyskretny latent), CVAE (warunkowy), hierarchiczne VAE. VAE jest też składnikiem Stable Diffusion (kompresja obrazu do latentu).

```python
def vae_loss(x_hat, x, mu, logvar):
    rec = F.mse_loss(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + kl
```

**Źródła:**
- [Outcome School - Variational Autoencoders](https://outcomeschool.com/blog/variational-autoencoders)
- [Kingma, Welling - Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Doersch - Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- [Kingma, Welling - An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691)

---

<a id="q98"></a>
### 98. W jaki sposób VAE narzuca probabilistyczną strukturę przestrzeni latentnej i dlaczego jest to ważne?

**Odpowiedź:**

VAE narzuca strukturę na trzy sposoby: przez **probabilistyczne kodowanie**, **człon KL** w funkcji celu i **reparametryzację**.

#### 1. Kodowanie jako rozkład
Enkoder zwraca parametry rozkładu `q_φ(z|x) = N(μ(x), diag σ²(x))`. Każdy przykład zajmuje „obłoczek” w latencie, a nie punkt. Dekoder musi poprawnie rekonstruować dane z **dowolnej** próbki z tego obłoku, więc sąsiednie punkty muszą dawać podobne wyjścia (szum wymusza gładkość).

#### 2. Regularyzacja KL do prioru
Człon $D_{KL}(q_\phi(z|x)\,\|\,\mathcal N(0,I))$ ściąga każdy posterior do standardowego rozkładu normalnego:
- `μ` blisko 0 (klastry nie uciekają w nieskończoność),
- `σ` blisko 1 (obłoczki nie zapadają się do punktów).

Bez KL enkoder mógłby zmniejszyć `σ -> 0`, dając zwykły autoenkoder. Z kolei tylko KL prowadziłoby do ignorowania danych. Optymalizacja równoważy oba człony: rekonstrukcja rozdziela różne przykłady, KL je zbliża i zaokrągla przestrzeń. Zagregowany posterior `q(z)=E_x q(z|x)` zbliża się do prioru `p(z)`.

#### 3. Reparametryzacja
$z=\mu+\sigma\odot\epsilon$, $\epsilon\sim\mathcal N(0,I)$ - losowość przeniesiona do zewnętrznego szumu, więc gradient przepływa przez `μ` i `σ` (niskowariancyjny estymator, zamiast REINFORCE).

#### Dlaczego to ważne
1. **Generowanie**: skoro `q(z)≈p(z)=N(0,I)`, możemy losować `z ~ N(0,I)` i dekodować w sensowne dane. W zwykłym AE nie wiemy, skąd próbkować, i łatwo trafić w „dziury”.
2. **Ciągłość i interpolacja**: bliskie punkty latentu dają podobne dane, interpolacje są płynne.
3. **Kompresja informacji**: KL działa jak kara za liczbę bitów informacji o `x` w `z` (kompromis rate-distortion), co zapobiega zapamiętywaniu.
4. **Rozplątanie**: z większą wagą `β` (β-VAE) wymiary latentu są bardziej niezależne, a interpretowalne.
5. **Podstawa probabilistyczna**: ELBO jest dolnym ograniczeniem `log p(x)`, więc trening ma sens statystyczny (wnioskowanie wariacyjne), a model pozwala szacować niepewność.

#### Pułapki
- Zbyt silne KL -> **posterior collapse** (latent nie niesie informacji, rozmyte wyniki). Mitigacje: KL annealing/warm-up, free bits, słabszy dekoder.
- Zbyt słabe KL -> nieregularny latent.
- Założenie gaussowskiego prioru może być zbyt ograniczające (alternatywy: VampPrior, normalizing flows, VQ-VAE).

**Źródła:**
- [Kingma, Welling - Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Doersch - Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
- [Higgins et al. - beta-VAE (ICLR 2017)](https://openreview.net/forum?id=Sy2fzU9gl)
- [Bowman et al. - Generating Sentences from a Continuous Space (posterior collapse, KL annealing)](https://arxiv.org/abs/1511.06349)

---

<a id="q99"></a>
### 99. Jaka jest architektura sieci GAN (Generative Adversarial Network)?

**Odpowiedź:**

**GAN** (Goodfellow i in., 2014) składa się z dwóch sieci trenowanych **przeciwstawnie** w grze o sumie zerowej:
- **Generator `G`** - mapuje losowy wektor szumu `z ~ p(z)` (np. `N(0, I)`) na sztuczną próbkę `G(z)`.
- **Dyskryminator `D`** - klasyfikator binarny, zwraca prawdopodobieństwo `D(x)`, że `x` pochodzi z danych rzeczywistych (a nie z generatora).

#### Cel minimax

$$\min_G \max_D \; V(D,G)=\mathbb{E}_{x\sim p_{data}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

Dla optymalnego `D` cel `G` sprowadza się do minimalizacji rozbieżności Jensena-Shannona między `p_data` a `p_g`. Równowaga Nasha: `p_g = p_data`, a `D(x) = 1/2`.

#### Schemat treningu (naprzemienny)
1. Pobierz batch prawdziwych danych i batch wygenerowanych `G(z)`.
2. Zaktualizuj **D** (gradient ascent na `V`): rozróżniaj prawdziwe od fałszywych.
3. Zaktualizuj **G**: w praktyce maksymalizuj `log D(G(z))` (tzw. *non-saturating loss*), bo oryginalna strata `log(1 - D(G(z)))` daje słaby gradient na początku, gdy `D` łatwo odrzuca próbki.
4. Powtarzaj.

#### Typowe architektury
- **MLP-GAN** - oryginalne, na MNIST.
- **DCGAN** - generator z transposed convolutions, BatchNorm, ReLU/tanh; dyskryminator z strided convolutions, LeakyReLU; brak warstw w pełni połączonych.
- **Conditional GAN (cGAN)** - `G(z|y)` i `D(x|y)` warunkowane etykietą/tekstem/obrazem.
- **WGAN / WGAN-GP** - odległość Wassersteina zamiast JS (stabilniejszy trening, gradient penalty).
- **Progressive GAN, StyleGAN** - generacja wysokiej rozdzielczości z kontrolą stylu.
- **Pix2Pix, CycleGAN** - tłumaczenie obraz-do-obrazu.

#### Uwagi praktyczne
- Trening jest niestabilny; wymaga zrównoważenia `G` i `D` (zbyt silny `D` -> zanik gradientu dla `G`).
- Metryki oceny: FID, Inception Score, precision/recall dla generatywnych.
- Techniki stabilizacji: spectral normalization, label smoothing, dwie skale learning rate (TTUR), augmentacja dyskryminatora.

```python
G = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())
D = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1))  # logit
```

**Źródła:**
- [Outcome School - Generative Adversarial Networks (GANs)](https://outcomeschool.com/blog/generative-adversarial-networks)
- [Goodfellow et al. - Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Radford et al. - Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434)
- [Goodfellow - NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)

---

<a id="q100"></a>
### 100. Jakie role pełnią generator i dyskryminator w sieci GAN?

**Odpowiedź:**

GAN to „gra” dwóch graczy: fałszerza (generator) i detektywa (dyskryminator). Analogia: fałszerz banknotów kontra policja - obaj doskonalą się, aż podróbki stają się nie do odróżnienia.

#### Generator `G`
- **Wejście:** wektor szumu `z` (opcjonalnie warunek `y`).
- **Wyjście:** próbka w przestrzeni danych (obraz, dźwięk...).
- **Cel:** oszukać dyskryminator, tzn. maksymalizować prawdopodobieństwo, że `D` uzna próbkę za prawdziwą: strata `L_G = -E[log D(G(z))]` (non-saturating).
- **Nigdy nie widzi prawdziwych danych** bezpośrednio, uczy się wyłącznie z gradientu przepływającego przez `D`.
- Architektura: sieci z transposed conv/upsampling (obrazy), Transformer/MLP (inne dane).

#### Dyskryminator `D`
- **Wejście:** próbka (prawdziwa lub wygenerowana).
- **Wyjście:** skalar (prawdopodobieństwo lub logit) „prawdziwa/fałszywa”.
- **Cel:** poprawnie klasyfikować: `L_D = -E[log D(x)] - E[log(1 - D(G(z)))]` (binary cross-entropy z etykietami 1 dla prawdziwych i 0 dla fałszywych).
- Architektura: klasyfikator (CNN ze strided conv, LeakyReLU). Po treningu zwykle jest odrzucany (w WGAN nazywany „krytykiem”, bo zwraca wynik bez interpretacji probabilistycznej).

#### Interakcja
- `D` dostarcza `G` sygnał uczący: mówi, **gdzie** próbki wyglądają niewiarygodnie.
- W miarę poprawy `G` zadanie `D` staje się trudniejsze, więc `D` musi uczyć się subtelniejszych cech.
- W idealnej równowadze `p_g = p_data`, a `D` zgaduje z prawdopodobieństwem 1/2.

#### Równowaga w praktyce
- `D` zbyt silny: `D(G(z)) ≈ 0`, gradient dla `G` zanika (dlatego non-saturating loss, WGAN).
- `G` zbyt silny lub `D` zbyt słaby: brak użytecznego sygnału.
- Często stosuje się kilka kroków `D` na 1 krok `G` (WGAN: 5:1), oddzielne optymalizatory (Adam z `β1=0.5`), TTUR.

```python
# krok dyskryminatora
d_loss = bce(D(real), ones) + bce(D(G(z).detach()), zeros)
# krok generatora
g_loss = bce(D(G(z)), ones)
```

Pamiętaj o `detach()` przy aktualizacji `D`, aby gradient nie płynął do `G`.

**Źródła:**
- [Outcome School - Generative Adversarial Networks (GANs)](https://outcomeschool.com/blog/generative-adversarial-networks)
- [Goodfellow et al. - Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- [Arjovsky et al. - Wasserstein GAN](https://arxiv.org/abs/1701.07875)
- [PyTorch - DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

---

<a id="q101"></a>
### 101. Czym jest zapadanie trybów (mode collapse) w GAN i jak można je łagodzić?

**Odpowiedź:**

**Mode collapse** to sytuacja, gdy generator produkuje tylko **wąski podzbiór** możliwych wyników (jeden lub kilka „trybów” rozkładu danych), ignorując resztę różnorodności. Np. GAN trenowany na MNIST generuje niemal wyłącznie cyfrę „1”, albo dla każdego `z` powstają niemal identyczne twarze.

#### Dlaczego się zdarza
- Generator jest nagradzany za **jakość** próbek (oszukanie `D`), nie za różnorodność. Znajduje kilka próbek, które `D` uznaje za realistyczne, i je powtarza.
- `D` ocenia próbki niezależnie, więc nie „widzi”, że wszystkie są podobne. Gdy `D` w końcu nauczy się odrzucać ten tryb, `G` przeskakuje do innego (oscylacja trybów).
- Nieodpowiednie kolejność/równowaga treningu, zbyt wysoki learning rate, niestabilna dynamika minimax; strata oryginalna jest kompatybilna z odwróceniem kolejności min/max (`G` może optymalizować względem chwilowego `D`).

#### Objawy
- Niska różnorodność próbek (wiele niemal identycznych obrazów).
- Strata generatora spada, jakość pojedynczych próbek dobra, ale słabe pokrycie rozkładu (niski recall, wysoki FID).
- Wizualizacja przestrzeni latentnej: różne `z` -> podobne wyjście.

#### Sposoby łagodzenia
| Technika | Idea |
|---|---|
| **WGAN / WGAN-GP** | odległość Wassersteina daje gradient nawet gdy rozkłady się nie pokrywają; stabilniejszy trening, mniej zapadania |
| **Unrolled GAN** | generator optymalizuje względem kilku przyszłych kroków `D` |
| **Minibatch discrimination / minibatch stddev** | `D` widzi statystyki całego batcha, więc wykrywa brak różnorodności (Salimans i in.; Progressive GAN) |
| **Feature matching** | `G` dopasowuje statystyki cech pośrednich `D`, a nie tylko wynik |
| **Regularyzacja `D`** | spectral normalization, gradient penalty, R1, dropout, szum w wejściu `D`, augmentacja dyskryminatora (ADA) |
| **Wiele generatorów / dyskryminatorów** | np. MGAN, mieszanka ekspertów |
| **Warunkowość** | cGAN (etykiety klas kierują różnorodnością) |
| **Lepsza optymalizacja** | TTUR (różne LR), Adam `β1=0.5`, więcej kroków `D`, label smoothing |
| **Inne cele** | PacGAN (`D` widzi paczki próbek), diversity-sensitive loss |
| **Alternatywne modele** | VAE, dyfuzja - z natury pokrywają rozkład szerzej |

#### Diagnostyka
FID, precision/recall dla rozkładów, Inception Score, liczba unikalnych trybów (w zadaniach syntetycznych, np. mieszanka 8 gaussów), inspekcja wizualna.

Mode collapse jest jednym z powodów, dla których GAN-y w wielu zastosowaniach ustąpiły modelom dyfuzyjnym.

**Źródła:**
- [Outcome School - Generative Adversarial Networks (GANs)](https://outcomeschool.com/blog/generative-adversarial-networks)
- [Salimans et al. - Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
- [Metz et al. - Unrolled Generative Adversarial Networks](https://arxiv.org/abs/1611.02163)
- [Gulrajani et al. - Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028)
- [Goodfellow - NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160)

---

<a id="q102"></a>
### 102. W jaki sposób GAN-y są używane w syntezie obrazów lub zadaniach tłumaczenia obraz-do-obrazu (image-to-image translation)?

**Odpowiedź:**

GAN-y uczą się generować obrazy o realistycznym rozkładzie. **Dyskryminator działa jak uczona funkcja straty** oceniająca „realizm”, co jest szczególnie cenne tam, gdzie prosta strata piksel-po-pikslu (L1/L2) daje rozmyte wyniki.

#### Synteza obrazów (z szumu lub warunku)
- **DCGAN** - pierwsze stabilne konwolucyjne GAN-y na obrazach.
- **Progressive GAN** - trening od niskiej do wysokiej rozdzielczości.
- **BigGAN** - duże GAN-y warunkowane klasą (ImageNet).
- **StyleGAN / StyleGAN2** - generator z modulacją stylu (AdaIN, mapowanie `z -> w`); fotorealistyczne twarze, kontrola cech na różnych poziomach szczegółowości, edycja w przestrzeni latentnej.
- **Text-to-image** (np. wczesne GAN-y warunkowane tekstem, GigaGAN); dziś dominują modele dyfuzyjne.

#### Image-to-image translation
1. **Pix2Pix** (Isola i in., 2017) - warunkowy GAN z **sparowanymi** danymi (np. szkic -> zdjęcie, mapa -> zdjęcie satelitarne, dzień -> noc). Generator: U-Net; dyskryminator: **PatchGAN** (ocenia realizm łatek N×N). Strata:

$$G^* = \arg\min_G\max_D\ \mathcal L_{cGAN}(G,D) + \lambda\,\mathcal L_{L1}(G)$$

   L1 zapewnia zgodność z docelowym obrazem, a GAN - ostrość i realizm.
2. **CycleGAN** (Zhu i in., 2017) - dla danych **niesparowanych** (konie <-> zebry, lato <-> zima). Dwa generatory `G: X->Y`, `F: Y->X` i dwa dyskryminatory. **Cycle-consistency loss**: $\|F(G(x))-x\|_1 + \|G(F(y))-y\|_1$ zapobiega dowolnym mapowaniom.
3. **StarGAN** - jedna sieć dla wielu domen (atrybuty twarzy).
4. **SPADE/GauGAN** - synteza z map semantycznych.

#### Inne zastosowania
- **Super-rozdzielczość** (SRGAN, ESRGAN) - ostrzejsze detale niż MSE.
- **Inpainting**, kolorowanie, usuwanie szumu i rozmycia.
- **Augmentacja danych** (np. obrazy medyczne przy rzadkich klasach) - z uwagą na walidację klinicznej wiarygodności.
- **Transfer stylu**, wirtualne przymierzanie, generowanie twarzy, deepfake (kwestie etyczne).
- Symulacja i adaptacja domeny (sim-to-real).

#### Uwagi i ograniczenia
- Trening niestabilny (mode collapse), potrzebne techniki stabilizacji (spectral norm, WGAN-GP).
- Możliwe artefakty i „halucynacje” cech, szczególnie w zastosowaniach wrażliwych (medycyna).
- Ocena: FID, LPIPS, SSIM/PSNR (gdy jest referencja), testy ludzkie.
- Obecnie wiele zadań (synteza z tekstu, edycja) przejęły modele dyfuzyjne (lepsza różnorodność i stabilność), choć GAN-y są wciąż atrakcyjne dla szybkiej, jednokrokowej generacji.

**Źródła:**
- [Isola et al. - Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix)](https://arxiv.org/abs/1611.07004)
- [Zhu et al. - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
- [Karras et al. - A Style-Based Generator Architecture for GANs (StyleGAN)](https://arxiv.org/abs/1812.04948)
- [Ledig et al. - Photo-Realistic Single Image Super-Resolution Using a GAN (SRGAN)](https://arxiv.org/abs/1609.04802)
- [Outcome School - Generative Adversarial Networks (GANs)](https://outcomeschool.com/blog/generative-adversarial-networks)

---

<a id="q103"></a>
### 103. Czym są konwolucyjne sieci neuronowe (Convolutional Neural Networks, CNN)?

**Odpowiedź:**

CNN to rodzina sieci neuronowych zaprojektowana do danych o strukturze siatki (obrazy, spektrogramy, szeregi czasowe 1D, woksele). Zamiast łączyć każdy neuron z każdym (jak w warstwie gęstej), CNN wykorzystuje **operację splotu** (w praktyce cross-correlation) z małymi, współdzielonymi filtrami przesuwanymi po wejściu.

#### Kluczowe idee
- **Lokalna łączność (local connectivity)** – neuron widzi tylko niewielkie okno (receptive field), bo piksele blisko siebie są silniej skorelowane.
- **Współdzielenie wag (weight sharing)** – ten sam filtr stosowany w każdym miejscu obrazu; to dramatycznie zmniejsza liczbę parametrów. Obraz 224×224×3 połączony gęsto z 1000 neuronami to ~150 mln wag, a filtr 3×3×3 ma ich 27 (+ bias).
- **Ekwiwariancja względem przesunięcia (translation equivariance)** – przesunięcie wejścia przesuwa mapę cech; wraz z poolingiem/globalnym uśrednianiem daje częściową inwariancję.
- **Hierarchia cech** – pierwsze warstwy uczą się krawędzi i tekstur, środkowe – fragmentów obiektów, głębokie – pojęć wysokiego poziomu.

#### Typowa architektura
`Conv -> BatchNorm -> ReLU -> (Pooling)` powtarzane wielokrotnie, potem global average pooling i warstwa gęsta z softmaxem (klasyfikacja). Historycznie: LeNet, AlexNet, VGG, ResNet (połączenia rezydualne), EfficientNet.

#### Splot 2D
Dla wejścia $X$ i filtra $K$ o rozmiarze $k\times k$:

$$Y[i,j] = \sum_{m=0}^{k-1}\sum_{n=0}^{k-1} K[m,n]\,X[i+m,\,j+n] + b$$

Dla wejścia z $C_{in}$ kanałami filtr ma wymiar $k\times k\times C_{in}$, a warstwa z $C_{out}$ filtrami ma $C_{out}(k^2C_{in}+1)$ parametrów.

```python
import torch.nn as nn
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10),
)
```

#### Zalety i ograniczenia
- Zalety: mało parametrów, silny bias indukcyjny dla obrazów, efektywne obliczenia na GPU.
- Ograniczenia: ograniczone pole recepcyjne (potrzeba głębokości lub dylatacji), słabsze modelowanie zależności globalnych niż attention, brak natywnej inwariancji na rotację/skalę (pomaga augmentacja). Vision Transformers (ViT) konkurują z CNN przy dużych zbiorach danych.

**Źródła:**
- [Stanford CS231n – Convolutional Neural Networks](https://cs231n.github.io/convolutional-networks/)
- [Deep Learning Book – Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
- [PyTorch – torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
- [Wikipedia – Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)

---

<a id="q104"></a>
### 104. Czym są filtry (kernels) w CNN?

**Odpowiedź:**

**Filtr (kernel)** to mały tensor uczonych wag (np. 3×3×C_in), który przesuwa się po wejściu i w każdej pozycji oblicza sumę ważoną – wynik tworzy jedną **mapę cech (feature map / activation map)**. Filtr działa jak detektor określonego wzorca: jeśli lokalny fragment wejścia jest do niego „podobny", aktywacja jest duża.

#### Co uczą się filtry
- Wagi filtrów nie są projektowane ręcznie (jak dawniej Sobel czy Gabor), lecz uczone przez backpropagation.
- Wczesne warstwy: krawędzie o różnych orientacjach, kolory, proste tekstury.
- Głębsze: fragmenty obiektów (oczy, koła), a na końcu całe koncepcje.

#### Wymiary i parametry
- Wejście: $H\times W\times C_{in}$. Warstwa z $C_{out}$ filtrami tworzy wyjście z $C_{out}$ kanałami – każdy filtr = jeden kanał wyjściowy.
- Liczba parametrów: $C_{out}\cdot(k_h k_w C_{in} + 1)$. Przykład: 64 filtry 3×3 na wejściu 32-kanałowym: $64\cdot(9\cdot32+1)=18\,496$.
- Filtr obejmuje **wszystkie** kanały wejściowe, sumuje po nich i dodaje jeden bias.

#### Rozmiar filtra
- 3×3 to standard (VGG): dwa stosy 3×3 mają pole recepcyjne 5×5, ale mniej parametrów ($2\cdot 9=18$ vs $25$ na parę kanałów) i więcej nieliniowości.
- 1×1 (pointwise) – miesza kanały bez zmiany rozdzielczości przestrzennej; używane do redukcji wymiarowości (bottleneck w ResNet, Inception).
- Warianty: dylatowane (dilated) zwiększają pole recepcyjne bez dodatkowych parametrów; depthwise separable (MobileNet) rozdzielają splot przestrzenny i mieszanie kanałów, zmniejszając koszt ok. $k^2$ razy w części przestrzennej.

#### Praktyka
- Inicjalizacja He/Kaiming dla ReLU.
- Wizualizacja filtrów pierwszej warstwy i map aktywacji pomaga w diagnostyce (np. „martwe" filtry).
- Pole recepcyjne rośnie z głębokością, stride i dylatacją.

**Źródła:**
- [Stanford CS231n – Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [Deep Learning Book – Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
- [MobileNets: Efficient CNNs for Mobile Vision Applications (arXiv)](https://arxiv.org/abs/1704.04861)
- [PyTorch – torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

---

<a id="q105"></a>
### 105. Czym jest stride w CNN?

**Odpowiedź:**

**Stride** (krok) określa, o ile pikseli filtr przesuwa się przy każdym kroku splotu. Stride = 1 oznacza przesuwanie co piksel (gęste próbkowanie), stride = 2 – co dwa piksele, co zmniejsza wynikową mapę cech mniej więcej dwukrotnie w każdym wymiarze.

#### Wzór na rozmiar wyjścia
Dla wejścia o rozmiarze $n$, filtra $k$, paddingu $p$ i stride'u $s$:

$$n_{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

Przykład: $n=32,\ k=3,\ p=1,\ s=2 \Rightarrow \lfloor (32+2-3)/2\rfloor+1 = 16$.

#### Efekty
- **Downsampling** – mniejsze mapy cech = mniej obliczeń (FLOPs) i pamięci; stride ≈ 4× mniej pikseli dla s=2 w 2D.
- **Szybszy wzrost pola recepcyjnego** w kolejnych warstwach.
- **Utrata informacji** – przy $s > 1$ część pozycji jest pomijana; przy $s > k$ filtr pomija piksele całkowicie (rzadko stosowane).
- Stride może zastąpić pooling: „striding convolution" uczy się własnego downsamplingu (np. ResNet używa stride=2 w konwolucjach; All-Convolutional Net eliminuje pooling).

#### Pułapki
- Duży stride w pierwszych warstwach ułatwia szybkie przetwarzanie, ale może gubić małe obiekty (istotne w detekcji).
- Stride 2 z kernelem niepodzielnym przez stride może powodować artefakty (checkerboard) w transponowanych konwolucjach (upsampling w dekoderach/GAN-ach).
- Rozmiar wyjścia nie zawsze jest liczbą całkowitą – framework zaokrągla w dół (floor).

```python
import torch, torch.nn as nn
x = torch.randn(1, 3, 32, 32)
print(nn.Conv2d(3, 8, 3, stride=2, padding=1)(x).shape)  # [1, 8, 16, 16]
```

**Źródła:**
- [Stanford CS231n – Convolutional Networks (spatial arrangement)](https://cs231n.github.io/convolutional-networks/)
- [A guide to convolution arithmetic for deep learning (arXiv)](https://arxiv.org/abs/1603.07285)
- [Striving for Simplicity: The All Convolutional Net (arXiv)](https://arxiv.org/abs/1412.6806)
- [PyTorch – torch.nn.Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

---

<a id="q106"></a>
### 106. Czym jest padding w CNN?

**Odpowiedź:**

**Padding** to dodanie dodatkowych wartości (najczęściej zer) wokół krawędzi wejścia przed operacją splotu. Bez paddingu każda konwolucja zmniejsza mapę cech o $k-1$ pikseli, a piksele brzegowe uczestniczą w mniejszej liczbie obliczeń niż środkowe.

#### Rodzaje
- **valid** – brak paddingu ($p=0$); wyjście mniejsze: $n-k+1$ (dla $s=1$).
- **same** – padding taki, że przy $s=1$ rozmiar wyjścia = rozmiar wejścia; dla nieparzystego $k$: $p=(k-1)/2$ (np. $k=3\Rightarrow p=1$, $k=5\Rightarrow p=2$).
- **full** – $p=k-1$, wyjście większe niż wejście (rzadziej w praktyce, używane np. w konwolucjach transponowanych).
- **Tryby wypełnienia**: zero padding (domyślny), reflect, replicate/edge, circular. Reflect/replicate zmniejszają artefakty brzegowe (np. w super-resolution, style transfer, segmentacji).

#### Po co
1. **Zachowanie rozmiaru przestrzennego** – umożliwia budowę bardzo głębokich sieci bez „kurczenia się" obrazu.
2. **Lepsze wykorzystanie krawędzi** – informacja z brzegów nie jest tracona.
3. **Kompatybilność wymiarów** – np. w połączeniach rezydualnych, które wymagają zgodnych kształtów.

#### Wzór
$$n_{out} = \left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1$$

#### Uwagi
- Zero padding wprowadza sztuczną „ramkę" i pozwala sieci (przez wielokrotne konwolucje) częściowo wnioskować o pozycji bezwzględnej – może to być korzystne lub szkodliwe (naruszenie ścisłej ekwiwariancji na przesunięcia).
- W 1D (tekst, audio) stosuje się analogiczne paddingi; w modelach kauzalnych (TCN, WaveNet) padding jest tylko z lewej strony, aby nie „widzieć przyszłości".

```python
nn.Conv2d(16, 32, kernel_size=3, padding=1, padding_mode="reflect")
```

**Źródła:**
- [A guide to convolution arithmetic for deep learning (arXiv)](https://arxiv.org/abs/1603.07285)
- [Stanford CS231n – Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
- [PyTorch – torch.nn.Conv2d (padding, padding_mode)](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)

---

<a id="q107"></a>
### 107. Czym jest pooling w CNN?

**Odpowiedź:**

**Pooling** to warstwa bez uczonych wag, która agreguje wartości w lokalnym oknie (np. 2×2) do jednej liczby, zmniejszając rozdzielczość map cech. Działa niezależnie na każdym kanale.

#### Rodzaje
- **Max pooling** – bierze maksimum z okna: zachowuje najsilniejszą aktywację (obecność cechy), najpopularniejszy w środku sieci.
- **Average pooling** – średnia z okna: gładsze wyjście.
- **Global average pooling (GAP)** – uśrednia całą mapę cech kanału do jednej liczby; zastępuje duże warstwy gęste na końcu (ResNet, Network in Network), zmniejsza overfitting i pozwala na wejścia o różnym rozmiarze.
- **Global max pooling**, adaptive pooling (wymusza zadany rozmiar wyjścia), RoI pooling (detekcja obiektów), spatial pyramid pooling.

#### Po co
- **Redukcja rozmiaru** – mniej obliczeń i pamięci (okno 2×2, stride 2 usuwa 75% aktywacji).
- **Częściowa inwariancja na małe przesunięcia** i deformacje.
- **Zwiększenie pola recepcyjnego** kolejnych warstw.
- **Regularyzacja** – mniej parametrów w dalszych warstwach.

#### Przykład
Wejście 4×4, okno 2×2, stride 2 -> wyjście 2×2. Dla bloku $\begin{pmatrix}1&3\\2&4\end{pmatrix}$ max pooling daje 4, average pooling 2,5.

#### Gradient
W max poolingu gradient płynie tylko do pozycji, która była maksimum (pozostałe dostają 0); w average poolingu rozkłada się równo na okno.

#### Wady i trend
- Utrata informacji przestrzennej (niekorzystna w segmentacji, detekcji małych obiektów).
- Współczesne architektury często zastępują pooling konwolucją ze stride=2 (np. ResNet po pierwszych warstwach), a na końcu stosują GAP.

```python
nn.MaxPool2d(kernel_size=2, stride=2)
nn.AdaptiveAvgPool2d((1, 1))
```

**Źródła:**
- [Stanford CS231n – Pooling Layer](https://cs231n.github.io/convolutional-networks/#pool)
- [Network In Network (arXiv)](https://arxiv.org/abs/1312.4400)
- [Deep Learning Book – Convolutional Networks (Pooling)](https://www.deeplearningbook.org/contents/convnets.html)
- [PyTorch – torch.nn.MaxPool2d](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html)

---

<a id="q108"></a>
### 108. Czym są warstwy w pełni połączone (fully connected layers) w CNN?

**Odpowiedź:**

**Warstwa w pełni połączona (FC / dense / linear)** łączy każdy neuron wejściowy z każdym neuronem wyjściowym: $y = \phi(Wx + b)$. W CNN pojawia się zwykle na końcu sieci, po blokach konwolucyjnych i poolingu.

#### Rola
- Bloki konwolucyjne wyodrębniają hierarchię cech przestrzennych. Mapa cech jest „spłaszczana" (flatten) do wektora, a warstwy FC łączą cechy globalnie i mapują je na wyjście zadania: logity klas (z softmaxem), wartość regresji, embedding.
- W klasycznych architekturach (AlexNet, VGG) ostatnie 2–3 warstwy FC po 4096 neuronów.

#### Problemy
- **Ogromna liczba parametrów**: w VGG-16 warstwy FC to ok. 120 z ~138 mln parametrów. Przykładowo flatten 7×7×512 = 25 088 -> 4096 daje ~103 mln wag.
- Skłonność do overfittingu (stąd dropout).
- Wymagają stałego rozmiaru wejścia (nie da się użyć obrazów o dowolnej rozdzielczości).
- Tracą strukturę przestrzenną.

#### Współczesne podejście
- **Global average pooling + jedna warstwa liniowa** (ResNet, Inception, EfficientNet) – dużo mniej parametrów, lepsza generalizacja.
- **Fully convolutional networks** – FC zamieniane na konwolucje 1×1, co pozwala na wejścia o dowolnym rozmiarze i gęste predykcje (segmentacja).
- W transfer learningu często wymienia się tylko końcową warstwę FC („głowę") na nową, dopasowaną do liczby klas.

```python
head = nn.Sequential(
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
    nn.Dropout(0.3), nn.Linear(512, num_classes)
)
```

**Źródła:**
- [Stanford CS231n – Convolutional Networks (Fully-connected layer)](https://cs231n.github.io/convolutional-networks/)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG, arXiv)](https://arxiv.org/abs/1409.1556)
- [Fully Convolutional Networks for Semantic Segmentation (arXiv)](https://arxiv.org/abs/1411.4038)
- [Deep Learning Book – Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)

---

<a id="q109"></a>
### 109. Czym jest rekurencyjna sieć neuronowa (Recurrent Neural Network, RNN)?

**Odpowiedź:**

RNN to sieć przetwarzająca **sekwencje** (tekst, audio, szeregi czasowe) krok po kroku, utrzymując **stan ukryty (hidden state)** $h_t$, który pełni rolę pamięci o dotychczasowym kontekście. Te same wagi są używane w każdym kroku czasowym (współdzielenie parametrów w czasie), więc sieć obsługuje sekwencje o dowolnej długości.

#### Równania (prosty RNN Elmana)
$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h),\qquad y_t = W_{hy}h_t + b_y$$

#### Tryby użycia
- **many-to-one** – klasyfikacja sentymentu (używamy ostatniego $h_T$),
- **one-to-many** – generowanie podpisów obrazów,
- **many-to-many** – tagowanie sekwencji (POS, NER), a także encoder–decoder (tłumaczenie, Seq2Seq),
- **bidirectional RNN** – dwa przebiegi (w przód i w tył), gdy dostępna jest cała sekwencja.

#### Trening
**Backpropagation Through Time (BPTT)** – rozwijamy sieć w czasie i propagujemy gradient wstecz przez wszystkie kroki; przy długich sekwencjach stosuje się **truncated BPTT**. Gradient względem $h_k$ zawiera iloczyn jakobianów $\prod_{t}\partial h_t/\partial h_{t-1}$, co jest źródłem problemów znikających/eksplodujących gradientów.

#### Zalety i wady
- Zalety: naturalny model sekwencji, stała liczba parametrów niezależnie od długości, pamięć O(1) w czasie inferencji (stały stan).
- Wady: przetwarzanie sekwencyjne (brak równoległości po czasie), trudność z długimi zależnościami, niestabilny trening.

Współcześnie zwykłe RNN zostały w większości zastąpione przez LSTM/GRU, a te – przez Transformery, choć rekurencja wraca w modelach stanowych (np. Mamba, RWKV).

```python
import torch.nn as nn
rnn = nn.RNN(input_size=32, hidden_size=64, num_layers=2, batch_first=True)
out, h_n = rnn(x)  # x: (batch, seq, 32)
```

**Źródła:**
- [Outcome School – Recurrent Neural Network](https://outcomeschool.com/blog/recurrent-neural-network)
- [Deep Learning Book – Sequence Modeling: Recurrent and Recursive Nets](https://www.deeplearningbook.org/contents/rnn.html)
- [Karpathy – The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [PyTorch – torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)

---

<a id="q110"></a>
### 110. Jakie są ograniczenia RNN i jak się je rozwiązuje?

**Odpowiedź:**

#### Główne ograniczenia
1. **Znikające gradienty (vanishing gradients)** – przy BPTT gradient jest mnożony przez jakobiany $W_{hh}^\top \mathrm{diag}(\phi')$; jeśli ich norma < 1, gradient maleje wykładowo z odległością. Sieć nie uczy się zależności długodystansowych.
2. **Eksplodujące gradienty (exploding gradients)** – norma > 1 powoduje gwałtowny wzrost gradientu, skoki loss, NaN.
3. **Brak równoległości** – $h_t$ zależy od $h_{t-1}$, więc trening nie skaluje się po osi czasu; wolne na GPU.
4. **Wąskie gardło stanu** – cała historia musi zmieścić się w wektorze o stałym rozmiarze (zwłaszcza w Seq2Seq bez attention).
5. **Zapominanie/ograniczona pamięć praktyczna** nawet przy teoretycznie nieograniczonym kontekście.

#### Rozwiązania
| Problem | Rozwiązanie |
|---|---|
| Vanishing gradient | **LSTM / GRU** (bramki i addytywna aktualizacja komórki), połączenia rezydualne/skip, ortogonalna inicjalizacja $W_{hh}$, ReLU/IRNN |
| Exploding gradient | **Gradient clipping** (po normie), regularyzacja, ostrożny learning rate |
| Wąskie gardło kontekstu | **Mechanizm attention** (Bahdanau, 2014) – dekoder patrzy na wszystkie stany enkodera |
| Brak równoległości, długie zależności | **Transformer** – self-attention łączy dowolne pozycje w jednym kroku, trening równoległy |
| Brak dostępu do przyszłości | Bidirectional RNN |
| Koszt BPTT | Truncated BPTT |

#### Uwagi
- Transformer ma koszt $O(n^2)$ w długości sekwencji, a RNN $O(n)$ i stały stan przy inferencji; stąd zainteresowanie modelami SSM/rekurencyjnymi (Mamba, RWKV) jako kompromisem.
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

**Źródła:**
- [Outcome School – How do RNNs and Transformers differ?](https://outcomeschool.com/blog/how-do-rnns-and-transformers-differ)
- [On the difficulty of training recurrent neural networks (arXiv)](https://arxiv.org/abs/1211.5063)
- [Neural Machine Translation by Jointly Learning to Align and Translate (arXiv)](https://arxiv.org/abs/1409.0473)
- [Attention Is All You Need (arXiv)](https://arxiv.org/abs/1706.03762)

---

<a id="q111"></a>
### 111. Czym są LSTM i GRU? Jak rozwiązują problem zależności długoterminowych?

**Odpowiedź:**

Obie architektury to rekurencyjne komórki z **bramkami (gates)**, które uczą się, co zapamiętać, co zapomnieć i co ujawnić na wyjściu.

#### LSTM (Hochreiter & Schmidhuber, 1997)
Oprócz stanu ukrytego $h_t$ ma **stan komórki (cell state) $c_t$** – „autostradę" informacji:

$$c_t = f_t\odot c_{t-1} + i_t\odot \tilde c_t,\qquad h_t = o_t\odot\tanh(c_t)$$

Kluczem jest **addytywna** aktualizacja $c_t$. Gradient wzdłuż $c$ to $\partial c_t/\partial c_{t-1}=f_t$ (plus człony pośrednie) – gdy bramka zapominania jest bliska 1, gradient płynie prawie bez tłumienia (tzw. constant error carousel), zamiast być wielokrotnie przemnażany przez macierz wag i pochodną tanh jak w zwykłym RNN.

#### GRU (Cho et al., 2014)
Uproszczenie: brak osobnego stanu komórki, dwie bramki:
- **update gate** $z_t$ – ile starego stanu zachować,
- **reset gate** $r_t$ – ile przeszłości wziąć pod uwagę przy kandydacie.

$$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$$

(w części implementacji role $z_t$ i $1-z_t$ są zamienione – to kwestia konwencji.)

#### Porównanie
| | LSTM | GRU |
|---|---|---|
| Stany | $h_t$, $c_t$ | tylko $h_t$ |
| Bramki | 3 (input, forget, output) | 2 (update, reset) |
| Parametry | ~4 macierze na warstwę | ~3 macierze (ok. 25% mniej) |
| Wydajność | Nieco lepszy dla bardzo długich sekwencji | Szybszy, zwykle porównywalna jakość |

W praktyce wybór zależy od zbioru danych; często warto sprawdzić oba. Nadal cierpią na brak równoległości po czasie, dlatego w NLP dominują Transformery; LSTM/GRU pozostają używane w małych modelach, szeregach czasowych i urządzeniach o ograniczonych zasobach.

```python
lstm = nn.LSTM(128, 256, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
gru  = nn.GRU(128, 256, batch_first=True)
```

Wskazówka: inicjalizacja biasu bramki forget wartością ~1 ułatwia początkowe zapamiętywanie.

**Źródła:**
- [Understanding LSTM Networks – Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Learning Phrase Representations using RNN Encoder-Decoder (GRU, arXiv)](https://arxiv.org/abs/1406.1078)
- [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling (arXiv)](https://arxiv.org/abs/1412.3555)
- [PyTorch – torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

---

<a id="q112"></a>
### 112. Jakie są główne bramki w LSTM i jakie pełnią role?

**Odpowiedź:**

Wejściem komórki w kroku $t$ są $x_t$ i $h_{t-1}$. Wszystkie bramki to warstwy sigmoidalne (wartości 0–1) sterujące przepływem informacji przez mnożenie element po elemencie.

#### 1. Bramka zapominania (forget gate)
$$f_t=\sigma(W_f[h_{t-1},x_t]+b_f)$$
Decyduje, jaką część poprzedniego stanu komórki $c_{t-1}$ zachować (1 = zachowaj, 0 = zapomnij). Np. po zakończeniu zdania sieć może „zapomnieć" rodzaj poprzedniego podmiotu.

#### 2. Bramka wejściowa (input gate) + kandydat
$$i_t=\sigma(W_i[h_{t-1},x_t]+b_i),\qquad \tilde c_t=\tanh(W_c[h_{t-1},x_t]+b_c)$$
$\tilde c_t$ to proponowana nowa treść pamięci, a $i_t$ mówi, ile z niej zapisać.

#### 3. Aktualizacja stanu komórki
$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

#### 4. Bramka wyjściowa (output gate)
$$o_t=\sigma(W_o[h_{t-1},x_t]+b_o),\qquad h_t=o_t\odot\tanh(c_t)$$
Określa, jaka część stanu komórki zostaje ujawniona jako stan ukryty i wyjście do następnej warstwy/kroku.

#### Intuicja
- $c_t$ – długoterminowa pamięć, $h_t$ – krótkoterminowy „widok" pamięci.
- Addytywna aktualizacja $c_t$ pozwala gradientowi płynąć bez wielokrotnego zmniejszania.

#### Warianty
- **Peephole connections** – bramki widzą także $c_{t-1}$.
- **Coupled forget/input** ($i_t = 1-f_t$) – prowadzi do rozwiązania zbliżonego do GRU.

#### Praktyka
- Cztery macierze wag są zwykle sklejane w jedną operację (stąd w PyTorch kolejność bramek `i, f, g, o` w `weight_ih`).
- Bias bramki forget inicjalizuje się na ~1.
- Sprawdzenie liczby parametrów: $4\,(d_h(d_x+d_h)+d_h)$ na warstwę jednokierunkową w matematycznym ujęciu (PyTorch ma dwa wektory biasu, więc $+d_h$ więcej na bramkę).

**Źródła:**
- [Understanding LSTM Networks – Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Wikipedia – Long short-term memory](https://en.wikipedia.org/wiki/Long_short-term_memory)
- [LSTM: A Search Space Odyssey (arXiv)](https://arxiv.org/abs/1503.04069)
- [PyTorch – torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)

---

<a id="q113"></a>
### 113. Jak zidentyfikować problem eksplodujących gradientów w modelu?

**Odpowiedź:**

**Eksplodujące gradienty** występują, gdy gradienty rosną wykładowo podczas propagacji wstecznej (typowo w głębokich sieciach i RNN), co destabilizuje trening.

#### Objawy
- **Loss skacze gwałtownie**, rośnie lub staje się `NaN`/`inf` (często nagle po okresie normalnego uczenia).
- **Wagi** przyjmują ogromne wartości lub NaN.
- **Norma gradientu** (globalna) rośnie o rzędy wielkości, z widocznymi skokami (spikes).
- Aktywacje bardzo duże, nasycone (np. sigmoid/tanh), przepełnienia w fp16.
- Model nie uczy się mimo zmniejszania loss w pojedynczych krokach; metryki oscylują.

#### Jak to zmierzyć
```python
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))
# zwraca globalną normę – loguj ją w każdym kroku (np. TensorBoard/W&B)
for n, p in model.named_parameters():
    if p.grad is not None:
        print(n, p.grad.norm().item())
```
- Loguj normy per warstwa (histogramy gradientów), sprawdź `torch.isnan(loss)`, użyj `torch.autograd.set_detect_anomaly(True)` do zlokalizowania źródła NaN.
- Porównaj normy gradientów w pierwszych i ostatnich warstwach: w RNN – wzdłuż osi czasu.

#### Typowe przyczyny
- Zbyt duży learning rate; brak warmupu (Transformery).
- Zła inicjalizacja wag; brak normalizacji (BatchNorm/LayerNorm).
- Głębokie sieci bez połączeń rezydualnych; RNN z $\|W_{hh}\|>1$.
- Nieustabilne funkcje straty (np. log(0)), niewyskalowane wejścia/targety, błędne dane (outliery).

#### Naprawa
- **Gradient clipping** (po normie, np. 1.0) – standard w RNN i LLM.
- Mniejszy LR, warmup i scheduler; optimizer z adaptacją (Adam) z rozsądnym epsilon.
- Właściwa inicjalizacja (He/Xavier), normalizacja warstw, połączenia rezydualne, LSTM/GRU zamiast prostego RNN.
- Standaryzacja danych, stabilne numerycznie funkcje (`log_softmax`, `BCEWithLogitsLoss`), mixed precision z loss scalingiem.

Uwaga: objawy podobne do znikających gradientów wyglądają odwrotnie – loss stoi w miejscu, gradienty w wczesnych warstwach ~0.

**Źródła:**
- [On the difficulty of training recurrent neural networks (arXiv)](https://arxiv.org/abs/1211.5063)
- [Deep Learning Book – Optimization for Training Deep Models (cliffs and exploding gradients)](https://www.deeplearningbook.org/contents/optimization.html)
- [PyTorch – torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html)
- [Wikipedia – Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)

---

<a id="q114"></a>
### 114. Czym jest architektura Transformer i co odróżnia ją od CNN i RNN?

**Odpowiedź:**

Transformer (Vaswani et al., 2017, „Attention Is All You Need") to architektura oparta wyłącznie na mechanizmie **self-attention** i warstwach feed-forward, bez rekurencji i splotów. Jest fundamentem BERT, GPT, T5, ViT i współczesnych LLM.

#### Budowa
- **Embeddingi tokenów + kodowanie pozycji** (sinusoidalne, uczone, RoPE, ALiBi) – ponieważ attention jest niewrażliwe na kolejność.
- **Blok enkodera**: Multi-Head Self-Attention -> Add & Norm -> FFN (dwie warstwy liniowe z nieliniowością) -> Add & Norm.
- **Blok dekodera**: masked self-attention (kauzalna), cross-attention do wyjść enkodera (w wersji encoder–decoder), FFN.
- Połączenia rezydualne i LayerNorm stabilizują trening głębokich stosów.
- Modele: encoder-only (BERT), decoder-only (GPT, Llama), encoder–decoder (T5, oryginalny Transformer).

#### Self-attention
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
Każdy token tworzy zapytanie $Q$, klucz $K$ i wartość $V$; wynik to średnia ważona wartości wszystkich tokenów, z wagami zależnymi od podobieństwa zapytania do kluczy. Wiele głowic uczy się różnych relacji.

#### Porównanie
| | CNN | RNN/LSTM | Transformer |
|---|---|---|---|
| Kontekst | lokalny (rośnie z głębokością) | sekwencyjny, przez stan | globalny w jednej warstwie |
| Ścieżka między odległymi tokenami | $O(n/k)$ warstw lub log | $O(n)$ kroków | $O(1)$ |
| Równoległość treningu | tak | nie (po czasie) | tak |
| Koszt w długości $n$ | $O(n)$ | $O(n)$ | $O(n^2)$ czasu i pamięci attention |
| Bias indukcyjny | lokalność, translacja | kolejność | słaby (dane + kodowanie pozycji) |

#### Konsekwencje
- Świetnie skaluje się z danymi i mocą obliczeniową (scaling laws), łatwe do wstępnego trenowania na ogromnych korpusach.
- Wady: kwadratowy koszt (rozwiązania: FlashAttention, sparse/linear attention, KV cache), wymaga dużo danych, brak wbudowanego bias na kolejność.

**Źródła:**
- [Outcome School – Decoding Transformer Architecture](https://outcomeschool.com/blog/decoding-transformer-architecture)
- [Attention Is All You Need (arXiv)](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Stanford CS224n – NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)

---

<a id="q115"></a>
### 115. Czym jest mechanizm uwagi (Attention) w deep learningu i dlaczego jest istotny?

**Odpowiedź:**

**Attention** pozwala modelowi dynamicznie ważyć, które części wejścia są istotne dla aktualnie liczonego wyjścia, zamiast kompresować całą informację w jednym wektorze o stałym rozmiarze.

#### Geneza
W Seq2Seq z RNN enkoder kodował całe zdanie w jednym wektorze, co było wąskim gardłem przy długich zdaniach. Bahdanau et al. (2014) wprowadzili attention: dekoder w każdym kroku oblicza wagi względem wszystkich stanów enkodera i buduje **wektor kontekstu** jako ich średnią ważoną.

#### Ogólna postać (Q, K, V)
- **Query** – czego szukam, **Key** – co oferuje każdy element, **Value** – zawartość do pobrania.
- Wagi: $\alpha_{ij}=\mathrm{softmax}_j(\mathrm{score}(q_i,k_j))$, wynik: $o_i=\sum_j\alpha_{ij}v_j$.
- Scaled dot-product: $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$. Skalowanie przez $\sqrt{d_k}$ zapobiega nasyceniu softmaxu przy dużych wymiarach.

#### Warianty
- **Additive (Bahdanau)** vs **multiplicative/dot-product (Luong, Transformer)**.
- **Self-attention** – Q, K, V z tej samej sekwencji (modelowanie zależności wewnątrz sekwencji).
- **Cross-attention** – Q z jednej sekwencji, K/V z innej (tłumaczenie, dyfuzja tekst->obraz).
- **Multi-head attention** – równoległe „głowice" w różnych podprzestrzeniach.
- **Masked (causal) attention** – zakaz patrzenia w przyszłość (LLM).
- Efektywne: FlashAttention, sparse, sliding window, grouped-query attention.

#### Dlaczego istotne
1. Rozwiązuje wąskie gardło kontekstu i ułatwia długie zależności (ścieżka $O(1)$).
2. Umożliwia pełną równoległość treningu (Transformer).
3. Daje częściową interpretowalność (mapy uwagi, choć nie zawsze są wiarygodnym wyjaśnieniem).
4. Uniwersalny: NLP, wizja (ViT), audio, biologia (AlphaFold), multimodalność.

#### Koszt
Standardowy attention ma koszt $O(n^2 d)$ i pamięć $O(n^2)$ w długości sekwencji.

```python
import torch, math
def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None: scores = scores.masked_fill(mask == 0, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
```

**Źródła:**
- [Neural Machine Translation by Jointly Learning to Align and Translate (arXiv)](https://arxiv.org/abs/1409.0473)
- [Attention Is All You Need (arXiv)](https://arxiv.org/abs/1706.03762)
- [Effective Approaches to Attention-based Neural Machine Translation (arXiv)](https://arxiv.org/abs/1508.04025)
- [Lilian Weng – Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)

---

<a id="q116"></a>
### 116. Jaka jest podstawowa różnica między LSTM a Transformerami?

**Odpowiedź:**

Podstawowa różnica: **LSTM przetwarza sekwencję krok po kroku, przekazując informację przez stan rekurencyjny, a Transformer przetwarza wszystkie pozycje naraz i łączy je bezpośrednio przez self-attention.**

| Aspekt | LSTM | Transformer |
|---|---|---|
| Przepływ informacji | sekwencyjnie przez $h_t, c_t$ | bezpośrednie połączenia każdy-z-każdym |
| Równoległość treningu | brak po osi czasu | pełna (po pozycjach) |
| Zależności dalekiego zasięgu | ścieżka $O(n)$, informacja może się „rozmywać" | ścieżka $O(1)$ |
| Kolejność | wbudowana (kolejność przetwarzania) | wymaga kodowania pozycji |
| Koszt obliczeń w $n$ | $O(n)$ | $O(n^2)$ (attention) |
| Pamięć przy inferencji | stały stan $O(1)$ | KV cache rośnie liniowo z $n$ |
| Skalowanie z danymi i mocą | ograniczone | bardzo dobre (scaling laws) |
| Dane wymagane | mniej, dobre w małych zbiorach | więcej (słabszy bias indukcyjny) |

#### Konsekwencje praktyczne
- Transformery dominują w NLP, wizji, audio, ponieważ trening na tysiącach GPU jest możliwy dzięki równoległości.
- LSTM: przydatne w streamingu, urządzeniach brzegowych, szeregach czasowych i przy małych danych, gdzie liczy się stały koszt inferencji.
- Transformer nie ma z definicji pamięci między oknami kontekstu – musi się zmieścić w oknie; LSTM teoretycznie przenosi stan dowolnie daleko, choć w praktyce zapomina.
- Trend: modele hybrydowe i rekurencyjne o liniowym koszcie (Mamba/SSM, RWKV), łączące zalety obu podejść.

**Źródła:**
- [Outcome School – How do RNNs and Transformers differ?](https://outcomeschool.com/blog/how-do-rnns-and-transformers-differ)
- [Attention Is All You Need (arXiv)](https://arxiv.org/abs/1706.03762)
- [Understanding LSTM Networks – Christopher Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv)](https://arxiv.org/abs/2312.00752)

---

<a id="q117"></a>
### 117. Czym są modele dyfuzyjne (Diffusion Models)?

**Odpowiedź:**

Modele dyfuzyjne to generatywne modele, które uczą się **odwracać stopniowy proces dodawania szumu** do danych. Generowanie polega na starcie od czystego szumu gaussowskiego i iteracyjnym „odszumianiu" do próbki z rozkładu danych. Dominują w generowaniu obrazów (Stable Diffusion, DALL-E 2/3, Imagen), a także wideo, audio i cząsteczek.

#### Proces w przód (forward)
Stały (nieuczony) łańcuch Markowa dodający szum wg harmonogramu $\beta_t$:
$$q(x_t\mid x_{t-1})=\mathcal N\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big)$$
Zamknięta postać: $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$, gdzie $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$, $\epsilon\sim\mathcal N(0,I)$. Po $T$ krokach (np. 1000) $x_T$ to prawie czysty szum.

#### Proces wsteczny (reverse)
Sieć (zwykle U-Net lub Transformer, DiT) uczy się przewidywać szum $\epsilon_\theta(x_t,t)$ (albo $x_0$, albo „v"). Uproszczona funkcja straty (DDPM):
$$L=\mathbb E_{x_0,\epsilon,t}\big[\|\epsilon-\epsilon_\theta(x_t,t)\|^2\big]$$

#### Próbkowanie
- DDPM: ~1000 kroków (wolne). DDIM i solvery ODE (DPM-Solver): 20–50 kroków. Destylacja/consistency models: 1–4 kroki.
- **Classifier-free guidance** – łączy predykcje warunkowe i bezwarunkowe, by zwiększyć wierność względem promptu kosztem różnorodności.
- **Latent diffusion** – dyfuzja w przestrzeni latentnej autoenkodera (VAE), dużo tańsza (Stable Diffusion).

#### Powiązania
Odpowiada uczeniu funkcji score $\nabla_x\log p_t(x)$ (score matching) i dyfuzji opisanej SDE.

#### Zalety i wady
- Zalety: stabilny trening (prosty regresyjny loss), wysoka jakość i różnorodność, brak mode collapse jak w GAN-ach, elastyczne warunkowanie (tekst, obraz, maska).
- Wady: wolne próbkowanie (wiele wywołań sieci), duży koszt treningu, trudniejsze zastosowanie do danych dyskretnych (tekst).

**Źródła:**
- [Outcome School – Diffusion Models](https://outcomeschool.com/blog/diffusion-models)
- [Denoising Diffusion Probabilistic Models (arXiv)](https://arxiv.org/abs/2006.11239)
- [High-Resolution Image Synthesis with Latent Diffusion Models (arXiv)](https://arxiv.org/abs/2112.10752)
- [Lilian Weng – What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

---

<a id="q118"></a>
### 118. Dlaczego dyfuzja działa lepiej niż autoregresja?

**Odpowiedź:**

Sformułowanie pytania jest skrótowe – **dyfuzja nie jest uniwersalnie lepsza**. Dla ciągłych, wielowymiarowych danych (obrazy, wideo, audio) dyfuzja zwykle wygrywa, a dla tekstu dominuje autoregresja (LLM). Poniżej powody i kontrargumenty.

#### Autoregresja (AR)
Rozkład faktoryzowany: $p(x)=\prod_i p(x_i\mid x_{<i})$. Generujemy element po elemencie w ustalonej kolejności.

#### Dlaczego dyfuzja często lepiej sprawdza się dla obrazów
1. **Brak sztucznej kolejności** – piksele nie mają naturalnej sekwencji; raster scan narzuca kolejność, a model musi przewidywać kolejne piksele bez globalnego planu. Dyfuzja poprawia cały obraz jednocześnie, od struktury globalnej (niska częstotliwość) do detali.
2. **Globalna spójność i możliwość „poprawek"** – każdy krok widzi cały obraz i może korygować wcześniejsze niedokładności; w AR błąd popełniony wcześniej nie może być poprawiony (błędy się kumulują, exposure bias).
3. **Praca na danych ciągłych** – nie wymaga dyskretyzacji (tokenizacji) z jej stratami, choć AR na tokenach VQ (np. Parti, DALL-E 1) też działa dobrze.
4. **Kompromis jakość/koszt sterowalny** – liczba kroków próbkowania i guidance pozwalają regulować jakość vs czas.
5. **Równoległość w obrębie kroku** – cała mapa jest przetwarzana naraz (choć kroków jest wiele).

#### Dlaczego AR nadal wygrywa dla tekstu
- Dane dyskretne o naturalnym porządku; dokładne likelihood, prosty trening (cross-entropy), skalowalność, KV cache, dojrzały ekosystem.
- Dyfuzja tekstu (masked/discrete diffusion) jest badana (np. LLaDA), obiecuje równoległe generowanie wielu tokenów i edycję w środku tekstu, lecz zwykle wymaga wielu iteracji i nie dorównuje jeszcze w pełni najlepszym AR w ekosystemie.

#### Koszt
- AR: $n$ sekwencyjnych wywołań modelu dla $n$ tokenów; dyfuzja: $T$ wywołań niezależnie od rozmiaru wyjścia (kilkadziesiąt).
- Trend: modele hybrydowe (AR + dyfuzja, np. w generacji wideo) i AR na obrazach z nowymi tokenizerami, które w niektórych pracach dorównują dyfuzji – wynik zależy od skali i implementacji.

**Źródła:**
- [Outcome School – Diffusion Models](https://outcomeschool.com/blog/diffusion-models)
- [Diffusion Models Beat GANs on Image Synthesis (arXiv)](https://arxiv.org/abs/2105.05233)
- [Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM, arXiv)](https://arxiv.org/abs/2107.03006)
- [Lilian Weng – What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

---

<a id="q119"></a>
### 119. Wyjaśnij transfer learning i kiedy go stosować.

**Odpowiedź:**

**Transfer learning** polega na wykorzystaniu wiedzy zdobytej przy rozwiązywaniu jednego zadania/na jednej dziedzinie (źródłowej, zwykle z dużą ilością danych) do poprawy uczenia na innym zadaniu (docelowym, często z małą ilością danych). Przykłady: model ImageNet użyty do klasyfikacji zdjęć RTG; BERT/LLM dostrojony do klasyfikacji zgłoszeń klientów.

#### Główne podejścia
1. **Feature extraction** – zamrożony pretrenowany backbone jako ekstraktor cech, trenujemy tylko nową „głowę" (np. regresja logistyczna na embeddingach). Szybkie, mało danych.
2. **Fine-tuning** – dalszy trening części lub wszystkich wag z małym learning rate (często discriminative LR: mniejszy dla niższych warstw; stopniowe odmrażanie – gradual unfreezing).
3. **Parameter-efficient fine-tuning (PEFT)** – LoRA, adaptery, prefix tuning: uczymy niewielką liczbę dodatkowych parametrów.
4. **Domain adaptation** – to samo zadanie, inny rozkład danych.
5. **Zero-/few-shot** – duże modele (LLM, CLIP) bez lub z minimalnym dostrajaniem.

#### Kiedy stosować
- Mało oznaczonych danych, a istnieje duży pretrenowany model w zbliżonej dziedzinie.
- Ograniczony budżet obliczeniowy/czas.
- Zadania podobne do źródłowych (obrazy naturalne, ten sam język).
- Chcemy lepszej generalizacji i szybszej zbieżności.

#### Kiedy ostrożnie
- Duża różnica domen (np. obrazy medyczne 3D vs ImageNet) – możliwy **negative transfer**; wtedy więcej fine-tuningu lub pretraining w domenie.
- Dużo własnych danych i pełna kontrola – trening od zera bywa równie dobry, ale kosztowniejszy.
- Różne modalności/tokenizery/licencje/prywatność modeli źródłowych.

#### Praktyczna reguła (wg CS231n)
| Mało danych | Dużo danych |
|---|---|
| Podobna domena: feature extraction | fine-tuning całości |
| Odległa domena: ostrożnie, trenuj więcej warstw | fine-tuning, ewentualnie od zera |

#### Pułapki
- Zbyt duży LR niszczy pretrenowane cechy (catastrophic forgetting).
- Zgodny preprocessing (normalizacja, tokenizer) z pretrainingiem.
- BatchNorm w małych batchach – rozważ tryb eval dla zamrożonych warstw.

```python
import torchvision, torch.nn as nn
m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
for p in m.parameters(): p.requires_grad = False
m.fc = nn.Linear(m.fc.in_features, num_classes)  # trenujemy tylko głowę
```

**Źródła:**
- [Stanford CS231n – Transfer Learning](https://cs231n.github.io/transfer-learning/)
- [How transferable are features in deep neural networks? (arXiv)](https://arxiv.org/abs/1411.1792)
- [A Survey on Transfer Learning (Pan & Yang, IEEE)](https://ieeexplore.ieee.org/document/5288526)
- [PyTorch – Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

<a id="q120"></a>
### 120. Czym są modele multimodalne (Multimodal AI) i jak przetwarzają różne typy danych?

**Odpowiedź:**

**Modele multimodalne** przyjmują i/lub generują dane z wielu modalności: tekst, obraz, dźwięk, wideo, dane tabelaryczne, czujniki. Przykłady: CLIP (obraz–tekst), GPT-4o, Gemini, Claude z obrazami (VLM – vision-language models), Whisper (mowa->tekst), modele text-to-image.

#### Jak przetwarzają dane
1. **Enkodery modalności** – każdy typ danych jest kodowany do wektorów: obraz -> ViT/CNN, audio -> spektrogram + encoder/Whisper, tekst -> tokenizer + embeddingi.
2. **Wspólna przestrzeń reprezentacji / projekcja** – wyniki enkoderów są rzutowane do przestrzeni wspólnej lub do przestrzeni embeddingów tokenów LLM (np. LLaVA: projekcja liniowa/MLP cech CLIP; Flamingo: cross-attention z resamplerem Perceiver).
3. **Fuzja (fusion)** – wczesna (surowe cechy łączone przed modelem), późna (osobne modele, łączenie predykcji) lub pośrednia (cross-attention, tokeny wizualne w sekwencji Transformera).
4. **Dekodowanie** – generowanie tekstu (dekoder LLM), obrazu (dyfuzja/AR na tokenach), audio.

#### Kluczowe techniki
- **Contrastive learning (CLIP)** – dwa enkodery uczone tak, by pasujące pary obraz–podpis miały bliskie embeddingi, a niepasujące – dalekie. Umożliwia zero-shot klasyfikację i wyszukiwanie.
$$\mathcal L=-\frac1N\sum_i \log\frac{\exp(s_{ii}/\tau)}{\sum_j\exp(s_{ij}/\tau)}$$
(symetrycznie w obu kierunkach; $s_{ij}$ – podobieństwo kosinusowe).
- **Natywne modele multimodalne** – jeden Transformer trenowany na mieszanych sekwencjach tokenów różnych modalności.
- **Instruction tuning** na danych obraz–tekst.

#### Zastosowania
Opis obrazów i VQA, OCR i analiza dokumentów, wyszukiwanie multimodalne, asystenci głosowi, robotyka, medycyna (obraz + raport).

#### Wyzwania
- Wyrównanie modalności (alignment), zbieranie parowanych danych, halucynacje wizualne, koszt obliczeń (wiele tokenów wizualnych), stronniczość, ewaluacja.

**Źródła:**
- [Outcome School – Multimodal AI](https://outcomeschool.com/blog/multimodal-ai)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP, arXiv)](https://arxiv.org/abs/2103.00020)
- [Visual Instruction Tuning (LLaVA, arXiv)](https://arxiv.org/abs/2304.08485)
- [Flamingo: a Visual Language Model for Few-Shot Learning (arXiv)](https://arxiv.org/abs/2204.14198)

---

<a id="q121"></a>
### 121. Jak działają modele świata (World Models)?

**Odpowiedź:**

**World model** to wyuczony model wewnętrznej dynamiki środowiska: na podstawie obserwacji i akcji przewiduje przyszłe stany (i nagrody). Agent może „wyobrażać sobie" konsekwencje działań, planować i uczyć się w symulacji zamiast kosztownych/niebezpiecznych interakcji ze światem.

#### Klasyczna architektura (Ha & Schmidhuber, 2018)
- **V (Vision)** – VAE kompresuje obserwację $o_t$ do niskowymiarowego wektora latentnego $z_t$.
- **M (Memory)** – RNN (MDN-RNN) przewiduje rozkład następnego stanu latentnego: $p(z_{t+1}\mid z_t,a_t,h_t)$.
- **C (Controller)** – mała sieć wybierająca akcję na podstawie $z_t$ i $h_t$.
Agent może być trenowany całkowicie „we śnie" (wewnątrz wyuczonego modelu).

#### Nowoczesne warianty
- **Dreamer (V1–V3)** – rekurencyjny model stanu przestrzeni latentnej (RSSM), agent uczy się polityki w wyobraźni (latent imagination) przez propagację gradientu przez dynamikę.
- **MuZero** – uczy się modelu tylko tych aspektów, które są istotne dla planowania (nagroda, wartość, polityka), z MCTS.
- **Generatywne world models wideo** (np. Genie, Sora traktowana jako symulator, Cosmos, V-JEPA) – uczą się przewidywać kolejne klatki/stany na dużych zbiorach wideo; **JEPA** przewiduje w przestrzeni reprezentacji, nie w pikselach.
- Zastosowania: robotyka, autonomiczna jazda, gry, planowanie, agent-based LLM.

#### Pętla działania
1. Zbieranie doświadczeń $(o_t,a_t,r_t,o_{t+1})$.
2. Uczenie modelu dynamiki (reprezentacja + predykcja) – strata rekonstrukcji/predykcji, KL w RSSM.
3. Planowanie lub trening polityki w wyobrażonych trajektoriach (rollouts).
4. Wdrożenie polityki, zbieranie nowych danych, powtórzenie.

#### Zalety i wyzwania
- Zalety: duża efektywność próbkowania (sample efficiency), bezpieczne uczenie, możliwość planowania.
- Wyzwania: **kumulacja błędów** modelu w długich rolloutach, model bias (agent wykorzystuje luki modelu), złożoność środowisk stochastycznych i częściowo obserwowalnych, wybór, co modelować (piksele czy abstrakcja), ewaluacja.

**Źródła:**
- [Outcome School – How do World Models work?](https://outcomeschool.com/blog/how-do-world-models-work)
- [World Models – Ha & Schmidhuber (arXiv)](https://arxiv.org/abs/1803.10122)
- [Mastering Diverse Domains through World Models (DreamerV3, arXiv)](https://arxiv.org/abs/2301.04104)
- [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero, arXiv)](https://arxiv.org/abs/1911.08265)

---

<a id="q122"></a>
### 122. Jak działają dyfuzyjne modele językowe (Diffusion Language Models, DLM)?

**Odpowiedź:**

**DLM** stosują ideę dyfuzji do tekstu: zamiast generować tokeny lewo->prawo, zaczynają od zaszumionej (zamaskowanej) sekwencji i w kilku iteracjach ją „odszumiają", przewidując wiele tokenów równolegle.

#### Problem: tekst jest dyskretny
Nie można dodać szumu gaussowskiego do tokenu. Rozwiązania:
1. **Dyfuzja w przestrzeni embeddingów** (np. Diffusion-LM) – szum na ciągłych embeddingach, potem zaokrąglanie do tokenów.
2. **Dyskretna dyfuzja** (D3PM) – proces w przód zamienia tokeny wg macierzy przejść, np. losowo lub na token `[MASK]`.
3. **Masked diffusion** – najpopularniejsza: proces w przód maskuje każdy token z prawdopodobieństwem $t\in[0,1]$; proces wsteczny uczy Transformera (dwukierunkowego, bez maski kauzalnej) przewidywać oryginalne tokeny na zamaskowanych pozycjach. Strata to (ważona) cross-entropy tylko na zamaskowanych tokenach – przypomina BERT/MLM, ale ze zmiennym poziomem maskowania.

#### Generowanie
1. Start: sekwencja samych `[MASK]` o zadanej długości (lub długość rozszerzana blokami).
2. W każdym kroku model przewiduje rozkłady dla wszystkich zamaskowanych pozycji.
3. Odmaskowuje się część tokenów (np. tych o największej pewności; strategia „low-confidence remasking"), resztę pozostawia jako maski.
4. Powtarza się aż do pełnego tekstu (liczba kroków $\ll$ liczba tokenów w idealnym przypadku).

#### Zalety
- **Równoległe generowanie** wielu tokenów -> potencjalnie mniejsze opóźnienie.
- **Dwukierunkowy kontekst** – możliwość uzupełniania w środku (infilling), kontrola struktury, poprawianie wcześniejszych fragmentów.
- Ciekawe własności, np. lepsze radzenie sobie z niektórymi zadaniami „odwróconymi" (reversal curse) raportowane w pracach o LLaDA.

#### Wady i otwarte problemy
- Jakość vs liczba kroków: mniej kroków = większa niespójność (tokeny wybierane niezależnie).
- Brak prostego KV cache jak w AR (attention dwukierunkowe) – trwają prace nad cache'owaniem i blokową dyfuzją.
- Trudniejsze estymowanie likelihood, dłuższy trening, mniej dojrzały ekosystem; wyniki zależą od skali – duże DLM (LLaDA, Mercury, Gemini Diffusion) są prezentowane jako konkurencyjne, ale AR nadal dominuje.

**Źródła:**
- [Outcome School – How do Diffusion Language Models (DLMs) work?](https://outcomeschool.com/blog/how-do-diffusion-language-models-dlms-work)
- [Large Language Diffusion Models (LLaDA, arXiv)](https://arxiv.org/abs/2502.09992)
- [Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM, arXiv)](https://arxiv.org/abs/2107.03006)
- [Diffusion-LM Improves Controllable Text Generation (arXiv)](https://arxiv.org/abs/2205.14217)

---

<a id="q123"></a>
### 123. Wyjaśnij Deep RL from Human Preferences (uczenie ze wzmocnieniem na podstawie ludzkich preferencji).

**Odpowiedź:**

Chodzi o podejście z pracy Christiano et al. (2017), *Deep Reinforcement Learning from Human Preferences*, które jest podstawą **RLHF** stosowanego przy dostrajaniu LLM (InstructGPT, ChatGPT).

#### Problem
W wielu zadaniach trudno zapisać funkcję nagrody (np. „napisz pomocną, bezpieczną odpowiedź"), a ręcznie zaprojektowana nagroda bywa nadużywana (reward hacking). Ludziom natomiast łatwo **porównać dwa wyniki** i wskazać lepszy.

#### Idea i pętla
1. Agent (polityka $\pi$) generuje trajektorie/odpowiedzi.
2. Człowiek dostaje pary fragmentów i wybiera preferowany.
3. **Model nagrody** $r_\theta$ jest uczony na tych porównaniach modelem Bradleya–Terry'ego:
$$P(\sigma^1\succ\sigma^2)=\frac{\exp\sum_t r_\theta(s^1_t,a^1_t)}{\exp\sum_t r_\theta(s^1_t,a^1_t)+\exp\sum_t r_\theta(s^2_t,a^2_t)}$$
z lossem cross-entropy względem etykiet ludzkich.
4. Polityka jest optymalizowana algorytmem RL (w pracy: A2C/TRPO; w LLM zwykle PPO) tak, by maksymalizować $r_\theta$.
5. Procedura jest powtarzana; nowe pary są wybierane m.in. tam, gdzie modele nagrody są niepewne (ensemble).

#### Zastosowanie w LLM (RLHF)
1. **SFT** – nadzorowane dostrojenie na demonstracjach.
2. **Trening modelu nagrody** na rankingach odpowiedzi.
3. **Optymalizacja PPO** z karą KL, by nie oddalić się zbytnio od modelu SFT:
$$\max_\pi\ \mathbb E[r_\theta(x,y)]-\beta\,\mathrm{KL}(\pi\,\|\,\pi_{ref})$$

#### Zalety
- Pozwala uczyć złożone, trudne do sformalizowania cele przy stosunkowo małej liczbie porównań (w pracy poniżej 1% interakcji agenta wymagało ludzkiej oceny).
- Lepsze dopasowanie do intencji użytkownika.

#### Wady i alternatywy
- Koszt i szum etykiet, stronniczość annotatorów, **reward hacking/over-optimization** modelu nagrody, niestabilność PPO.
- Alternatywy: **DPO** (bezpośrednia optymalizacja preferencji bez jawnego modelu nagrody i RL), RLAIF/Constitutional AI (preferencje generowane przez AI), GRPO.

**Źródła:**
- [Outcome School – Decoding Deep RL from Human Preferences](https://outcomeschool.com/blog/decoding-deep-rl-from-human-preferences)
- [Deep Reinforcement Learning from Human Preferences (arXiv)](https://arxiv.org/abs/1706.03741)
- [Training language models to follow instructions with human feedback (InstructGPT, arXiv)](https://arxiv.org/abs/2203.02155)
- [Direct Preference Optimization (arXiv)](https://arxiv.org/abs/2305.18290)

---

## NLP

<a id="q124"></a>
### 124. Jakie są zalety Transformerów nad tradycyjnymi modelami sequence-to-sequence?

**Odpowiedź:**

Tradycyjne Seq2Seq (Sutskever et al., 2014) to enkoder RNN/LSTM kompresujący wejście do wektora i dekoder RNN generujący wyjście. Transformer zastępuje rekurencję self-attention.

#### Główne zalety
1. **Równoległość treningu** – wszystkie pozycje sekwencji są przetwarzane jednocześnie (macierzowe mnożenia), a RNN muszą iść krok po kroku. Daje to znacznie szybszy trening na GPU/TPU i możliwość skalowania do miliardów parametrów.
2. **Zależności dalekiego zasięgu** – dowolne dwa tokeny łączy ścieżka długości $O(1)$ (vs $O(n)$ w RNN), więc gradient płynie krótszymi drogami i unika się problemu znikających gradientów.
3. **Brak wąskiego gardła kontekstu** – dekoder ma dostęp do reprezentacji wszystkich tokenów wejścia przez cross-attention, a nie do jednego wektora.
4. **Lepsza jakość** – w oryginalnej pracy rekordowe wyniki BLEU w tłumaczeniu WMT 2014 EN-DE i EN-FR przy mniejszym koszcie treningu.
5. **Multi-head attention** – wiele „perspektyw" relacji (składnia, koreferencja, semantyka).
6. **Transfer learning i skalowanie** – pretrenowanie na ogromnych korpusach (BERT, GPT, T5) i dostrajanie do zadań; przewidywalne scaling laws.
7. **Uniwersalność** – ta sama architektura działa dla tekstu, obrazów (ViT), audio, białek.
8. **Lepsza interpretowalność wizualna** (mapy uwagi – z zastrzeżeniami).

#### Koszty i kompromisy
- Attention ma koszt $O(n^2)$ czasu i pamięci, RNN – $O(n)$.
- Potrzeba więcej danych (słabszy bias indukcyjny) oraz kodowania pozycji.
- Inferencja autoregresyjna dekodera jest wciąż sekwencyjna (KV cache przyspiesza).

**Źródła:**
- [Outcome School – Decoding Transformer Architecture](https://outcomeschool.com/blog/decoding-transformer-architecture)
- [Attention Is All You Need (arXiv)](https://arxiv.org/abs/1706.03762)
- [Sequence to Sequence Learning with Neural Networks (arXiv)](https://arxiv.org/abs/1409.3215)
- [The Illustrated Transformer – Jay Alammar](https://jalammar.github.io/illustrated-transformer/)

---

<a id="q125"></a>
### 125. Jakie są ograniczenia Transformerów i jak można je adresować?

**Odpowiedź:**

| Ograniczenie | Opis | Sposoby zaradcze |
|---|---|---|
| **Kwadratowy koszt attention** $O(n^2)$ | pamięć i czas rosną z kwadratem długości sekwencji | FlashAttention (I/O-aware, dokładne), sparse/sliding-window attention (Longformer, BigBird), linear attention, Reformer, modele SSM (Mamba) |
| **Ograniczone okno kontekstu** | model widzi tylko $n$ tokenów | rozszerzanie pozycji (RoPE scaling, ALiBi), pamięć zewnętrzna, RAG, Transformer-XL (rekurencja segmentów), chunking |
| **Pamięć KV cache przy inferencji** | rośnie liniowo z długością i liczbą warstw/głowic | Multi-Query/Grouped-Query Attention, kwantyzacja KV, PagedAttention (vLLM), eviction |
| **Wielkie zapotrzebowanie na dane i obliczenia** | trening od zera bardzo drogi | transfer learning, PEFT (LoRA), destylacja, kwantyzacja, MoE (rzadka aktywacja) |
| **Wolne generowanie autoregresyjne** | token po tokenie | speculative decoding, batching ciągły, dyfuzja tekstu |
| **Halucynacje** | wiarygodne, lecz fałszywe treści | RAG, fine-tuning z RLHF/DPO, weryfikacja, grounding, narzędzia |
| **Słaba interpretowalność i niezawodność** | trudno wyjaśnić decyzje | analiza uwagi/aktywacji, mechanistic interpretability, testy |
| **Brak wbudowanej informacji o kolejności** | attention jest permutacyjnie ekwiwariantne | kodowanie pozycji (sinusoidalne, uczone, RoPE) |
| **Bias i toksyczność, wyciek danych** | odziedziczone po danych | filtrowanie, alignment, ewaluacje, red-teaming |
| **Słabe uogólnienie na dłuższe sekwencje niż w treningu** | ekstrapolacja pozycyjna | RoPE/YaRN, ALiBi, długokontekstowy fine-tuning |

#### Uwagi praktyczne
- **FlashAttention** nie zmienia matematyki, tylko układ obliczeń w pamięci GPU – zmniejsza pamięć do $O(n)$ i przyspiesza, ale czas nadal $O(n^2)$.
- Aproksymacje (sparse/linear) mogą obniżać jakość przy zadaniach wymagających precyzyjnego odczytu z kontekstu.
- Długi kontekst nie oznacza, że model wykorzystuje go równomiernie („lost in the middle").

**Źródła:**
- [FlashAttention: Fast and Memory-Efficient Exact Attention (arXiv)](https://arxiv.org/abs/2205.14135)
- [Longformer: The Long-Document Transformer (arXiv)](https://arxiv.org/abs/2004.05150)
- [Lilian Weng – The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)
- [Efficient Memory Management for LLM Serving with PagedAttention (arXiv)](https://arxiv.org/abs/2309.06180)

---

<a id="q126"></a>
### 126. Czym jest BERT i jak poprawia rozumienie języka?

**Odpowiedź:**

**BERT** (Bidirectional Encoder Representations from Transformers, Devlin et al., 2018) to pretrenowany **enkoder** Transformera, który tworzy kontekstowe reprezentacje tokenów, uwzględniając kontekst zarówno z lewej, jak i z prawej strony. Po dostrojeniu ustanowił rekordy w benchmarkach NLU (GLUE, SQuAD).

#### Architektura
Stos enkoderów Transformera: BERT-base (12 warstw, 768 wymiarów, 12 głowic, ~110 mln parametrów), BERT-large (24 warstwy, 1024, 16 głowic, ~340 mln). Wejście: tokeny WordPiece + `[CLS]` na początku + `[SEP]` między zdaniami; embeddingi = token + segment + pozycja.

#### Pretraining (dwa zadania)
1. **Masked Language Modeling (MLM)** – losowo wybiera się ~15% tokenów; z nich w 80% zamiana na `[MASK]`, w 10% na losowy token, w 10% bez zmian. Model przewiduje oryginał. Pozwala uczyć się głębokiej dwukierunkowości (zwykły LM jest jednokierunkowy).
2. **Next Sentence Prediction (NSP)** – czy zdanie B następuje po A (późniejsze prace, np. RoBERTa, pokazały, że NSP daje niewiele lub nic).

#### Dlaczego poprawia rozumienie języka
- **Kontekstowe embeddingi** – „bank" w „river bank" i „bank account" ma różne wektory (w przeciwieństwie do word2vec/GloVe).
- **Dwukierunkowość** – rozumienie zdania w całości.
- **Transfer learning** – jedna sieć pretrenowana na dużych korpusach (BooksCorpus, Wikipedia) i dostrajana z dodatkową małą głową do: klasyfikacji (`[CLS]`), NER (etykieta na token), QA (zakres odpowiedzi), NLI, semantic similarity.

#### Ograniczenia
- Nie generuje tekstu w naturalny sposób (brak dekodera).
- Maksymalna długość 512 tokenów.
- Niedopasowanie pretraining–fine-tuning z powodu tokenu `[MASK]`.
- Następcy: RoBERTa, ALBERT, DistilBERT, DeBERTa, ELECTRA, ModernBERT.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
out = model(**tok("Great movie!", return_tensors="pt"))
```

**Źródła:**
- [BERT: Pre-training of Deep Bidirectional Transformers (arXiv)](https://arxiv.org/abs/1810.04805)
- [Hugging Face – BERT model documentation](https://huggingface.co/docs/transformers/model_doc/bert)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (arXiv)](https://arxiv.org/abs/1907.11692)
- [The Illustrated BERT, ELMo, and co. – Jay Alammar](https://jalammar.github.io/illustrated-bert/)

---

<a id="q127"></a>
### 127. Jak trenuje się Transformery (pretraining i fine-tuning)?

**Odpowiedź:**

Standardowy paradygmat to **pretraining na ogromnych, nieoznakowanych danych**, a następnie **adaptacja** do konkretnych zadań.

#### 1. Pretraining (self-supervised)
Model uczy się reprezentacji języka bez ręcznych etykiet; etykiety pochodzą z samych danych.
- **Causal LM (GPT, Llama)** – przewidywanie następnego tokenu: $\mathcal L=-\sum_t\log p_\theta(x_t\mid x_{<t})$.
- **Masked LM (BERT)** – przewidywanie zamaskowanych tokenów.
- **Denoising / span corruption (T5, BART)** – odtwarzanie zniekształconego tekstu.
- Dane: setki miliardów–biliony tokenów (web, książki, kod) po deduplikacji i filtrowaniu; tokenizacja BPE/WordPiece/SentencePiece.
- Techniki: AdamW, warmup + cosine decay, mixed precision (bf16), gradient clipping, gradient accumulation, równoległość (data/tensor/pipeline, ZeRO/FSDP), checkpointing aktywacji.

#### 2. Fine-tuning
- **Supervised fine-tuning (SFT)** – na zbiorze zadaniowym (klasyfikacja, QA, NER) lub instrukcjach (instruction tuning) z małym learning rate (np. $10^{-5}$–$5\cdot10^{-5}$ dla BERT), kilka epok.
- **Dodanie głowy** zadaniowej (np. warstwa liniowa na `[CLS]`).
- **PEFT** – LoRA/QLoRA, adaptery – trening ułamka parametrów, mniej pamięci i ryzyka catastrophic forgetting.
- **Alignment** – RLHF/DPO dopasowujące do preferencji ludzi.

#### Typowy przebieg dla LLM
Pretraining -> SFT (instrukcje) -> preference tuning (RLHF/DPO) -> ewaluacja i bezpieczeństwo.

#### Praktyka i pułapki
- Ten sam tokenizer co w pretrainingu.
- Monitorowanie loss/perplexity walidacyjnej, early stopping, ewaluacja na zestawach zadaniowych.
- Ryzyko overfittingu na małym zbiorze i catastrophic forgetting (mniejszy LR, mieszanie danych ogólnych, LoRA).
- Scaling laws (Kaplan; Chinchilla) wskazują na proporcję rozmiaru modelu do liczby tokenów.

```python
from transformers import Trainer, TrainingArguments
args = TrainingArguments("out", learning_rate=2e-5, num_train_epochs=3,
                         per_device_train_batch_size=16, warmup_ratio=0.1, weight_decay=0.01)
Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds).train()
```

**Źródła:**
- [BERT: Pre-training of Deep Bidirectional Transformers (arXiv)](https://arxiv.org/abs/1810.04805)
- [Language Models are Few-Shot Learners (GPT-3, arXiv)](https://arxiv.org/abs/2005.14165)
- [Exploring the Limits of Transfer Learning with T5 (arXiv)](https://arxiv.org/abs/1910.10683)
- [Hugging Face – Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training)

---

<a id="q128"></a>
### 128. Wyjaśnij transfer learning w kontekście Transformerów.

**Odpowiedź:**

W NLP transfer learning oznacza, że model Transformera najpierw uczy się ogólnej wiedzy o języku (składnia, semantyka, wiedza o świecie) na ogromnym korpusie, a następnie jest **adaptowany** do konkretnego zadania z małą ilością danych. Zastąpił on wcześniejsze podejście: statyczne embeddingi (word2vec/GloVe) + model uczony od zera.

#### Etapy
1. **Pretraining** – self-supervised (MLM lub causal LM) na terabajtach tekstu. Kosztowny, wykonywany raz, modele udostępniane (Hugging Face Hub).
2. **Adaptacja** do zadania docelowego:
   - **Fine-tuning pełny** – wszystkie wagi + głowa zadaniowa; najlepsza jakość przy dużych danych.
   - **Feature-based** – zamrożone embeddingi/hidden states jako cechy dla prostego klasyfikatora.
   - **PEFT (LoRA, adapters, prompt/prefix tuning)** – trening niewielkiej liczby parametrów; łatwe utrzymanie wielu zadań na jednym modelu bazowym.
   - **In-context learning** – bez zmiany wag: kilka przykładów w prompcie (few-shot) lub sama instrukcja (zero-shot).
   - **Instruction tuning / alignment** – ogólny transfer do wielu nieznanych zadań.

#### Dlaczego działa
Reprezentacje z niższych warstw są ogólne (składnia, morfologia), wyższe – bardziej zadaniowe; pretraining daje dobrą inicjalizację i redukuje potrzebę danych oznakowanych o rzędy wielkości.

#### Kiedy stosować i uwagi
- Zawsze, gdy dostępny jest model dla języka/dziedziny; przy specyficznej dziedzinie (medycyna, prawo) – **domain-adaptive pretraining** (np. BioBERT, LegalBERT), a potem fine-tuning (ULMFiT: stopniowe odmrażanie i discriminative LR).
- Ryzyka: catastrophic forgetting, dziedziczone uprzedzenia, niedopasowanie domeny, licencje.
- Wybór modelu: encoder (BERT) dla klasyfikacji/NER/wyszukiwania, decoder (GPT/Llama) dla generacji, encoder–decoder (T5) dla tłumaczenia/streszczeń.

**Źródła:**
- [Universal Language Model Fine-tuning for Text Classification (ULMFiT, arXiv)](https://arxiv.org/abs/1801.06146)
- [BERT: Pre-training of Deep Bidirectional Transformers (arXiv)](https://arxiv.org/abs/1810.04805)
- [LoRA: Low-Rank Adaptation of Large Language Models (arXiv)](https://arxiv.org/abs/2106.09685)
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course)

---

<a id="q129"></a>
### 129. Opisz proces generowania tekstu w modelach językowych opartych na Transformerach.

**Odpowiedź:**

Modele decoder-only (GPT, Llama) generują tekst **autoregresyjnie**: przewidują rozkład następnego tokenu, wybierają token, doklejają go do kontekstu i powtarzają.

#### Kroki
1. **Tokenizacja** promptu (BPE/SentencePiece) -> ID tokenów, embeddingi + kodowanie pozycji.
2. **Przebieg przez stos dekoderów** (masked self-attention + FFN); na ostatniej pozycji otrzymujemy wektor ukryty.
3. **Logity** – rzut liniowy na słownik ($|V|$ rzędu 30k–200k), następnie softmax z temperaturą $T$:
$$p(x_t=i\mid x_{<t})=\frac{\exp(z_i/T)}{\sum_j\exp(z_j/T)}$$
4. **Dekodowanie** – wybór tokenu (patrz niżej).
5. **Doklejenie tokenu**, aktualizacja kontekstu i powtórzenie do tokenu końca sekwencji `<eos>` lub limitu długości.
6. **Detokenizacja** do tekstu.

#### Strategie dekodowania
- **Greedy** – zawsze najbardziej prawdopodobny token; deterministyczny, skłonny do powtórzeń.
- **Beam search** – utrzymuje $k$ najlepszych hipotez; dobre w tłumaczeniu/streszczaniu, mniej w otwartej generacji (nudne, powtarzalne).
- **Sampling** z **temperaturą** (niska T = bardziej deterministycznie), **top-k** (tylko $k$ najlepszych), **top-p / nucleus** (najmniejszy zbiór o łącznym prawdopodobieństwie $\ge p$), kary za powtórzenia.
- Contrastive search, speculative decoding (przyspieszenie z małym modelem roboczym).

#### Wydajność
- **KV cache** – klucze i wartości poprzednich tokenów są zapamiętywane, więc każdy nowy token wymaga liczenia tylko dla siebie (koszt kroku ~$O(n)$ zamiast $O(n^2)$).
- Faza **prefill** (równoległe przetwarzanie promptu) vs **decode** (token po tokenie, ograniczony przepustowością pamięci).
- Batching ciągły, kwantyzacja, PagedAttention.

#### Encoder–decoder
Enkoder koduje wejście raz; dekoder w każdym kroku używa cross-attention do wyjść enkodera.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2"); m = AutoModelForCausalLM.from_pretrained("gpt2")
ids = tok("The future of AI is", return_tensors="pt")
out = m.generate(**ids, max_new_tokens=30, do_sample=True, top_p=0.9, temperature=0.8)
print(tok.decode(out[0]))
```

**Źródła:**
- [Outcome School – Decoding Transformer Architecture](https://outcomeschool.com/blog/decoding-transformer-architecture)
- [The Curious Case of Neural Text Degeneration (nucleus sampling, arXiv)](https://arxiv.org/abs/1904.09751)
- [Hugging Face – Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [Attention Is All You Need (arXiv)](https://arxiv.org/abs/1706.03762)

---

<a id="q130"></a>
### 130. Czym są modele Seq2Seq?

**Odpowiedź:**

**Seq2Seq (sequence-to-sequence)** to modele mapujące sekwencję wejściową na sekwencję wyjściową, których długości mogą się różnić. Zastosowania: tłumaczenie maszynowe, streszczanie, dialog, rozpoznawanie mowy, generowanie kodu, text-to-SQL, korekta gramatyczna.

#### Architektura encoder–decoder (Sutskever et al., 2014; Cho et al., 2014)
- **Enkoder** (RNN/LSTM/GRU, dziś Transformer) czyta wejście $x_1..x_n$ i tworzy reprezentację (w klasycznej wersji – wektor kontekstu $c=h_n$).
- **Dekoder** generuje wyjście $y_1..y_m$ autoregresyjnie:
$$p(y\mid x)=\prod_{t=1}^{m}p(y_t\mid y_{<t},c)$$
- Start `<sos>`, koniec `<eos>`.

#### Trening i inferencja
- **Teacher forcing** – w treningu dekoder dostaje prawdziwy poprzedni token; przyspiesza uczenie, ale powoduje **exposure bias** (w inferencji widzi własne, potencjalnie błędne tokeny). Łagodzenie: scheduled sampling, RL.
- Loss: cross-entropy na każdym kroku.
- Inferencja: greedy lub **beam search**.

#### Ograniczenia klasycznego Seq2Seq
- **Wąskie gardło**: cała informacja w jednym wektorze -> pogorszenie dla długich zdań.
- Znikające gradienty w RNN, brak równoległości.

#### Ulepszenia
- **Attention (Bahdanau, Luong)** – dekoder dynamicznie patrzy na wszystkie stany enkodera.
- **Transformer** (2017) – encoder–decoder oparty na attention; T5, BART, mBART, MarianMT, Whisper.
- Mechanizmy pointer/copy, coverage; subword tokenization dla nieznanych słów (OOV).

#### Metryki
BLEU (tłumaczenie), ROUGE (streszczanie), WER (mowa), chrF, BERTScore, ocena ludzka.

**Źródła:**
- [Sequence to Sequence Learning with Neural Networks (arXiv)](https://arxiv.org/abs/1409.3215)
- [Learning Phrase Representations using RNN Encoder-Decoder (arXiv)](https://arxiv.org/abs/1406.1078)
- [Neural Machine Translation by Jointly Learning to Align and Translate (arXiv)](https://arxiv.org/abs/1409.0473)
- [Wikipedia – Seq2seq](https://en.wikipedia.org/wiki/Seq2seq)

---

<a id="q131"></a>
### 131. Porównaj modele N-gram i modele deep learningowe (kompromisy).

**Odpowiedź:**

**N-gram** to statystyczny model języka szacujący $P(w_t\mid w_{t-n+1..t-1})$ z zliczeń w korpusie. **Modele deep learningowe** (RNN/LSTM, Transformer) uczą się rozkładu za pomocą sieci neuronowych z reprezentacjami rozproszonymi.

| Cecha | N-gram | Deep learning |
|---|---|---|
| Kontekst | stałe okno $n-1$ słów | długi (LSTM) lub całe okno (Transformer) |
| Reprezentacja słów | dyskretne symbole | embeddingi; podobieństwo semantyczne |
| Generalizacja | słaba dla nieobserwowanych n-gramów | dobra dzięki embeddingom |
| Rzadkość danych | **rzadkość** rośnie wykładowo z $n$ ($|V|^n$ kombinacji); potrzeba wygładzania (Laplace, Kneser–Ney, backoff) | brak problemu z rzadkością w tej postaci |
| Dane potrzebne | mało do umiarkowanie | dużo |
| Koszt treningu | bardzo niski (zliczanie) | wysoki (GPU) |
| Inferencja | bardzo szybka (lookup) | kosztowniejsza |
| Interpretowalność | wysoka (tablice zliczeń) | niska |
| Pamięć | rośnie z liczbą unikalnych n-gramów | stała liczba parametrów |
| Jakość (perplexity, zadania) | znacznie niższa | znacznie wyższa |

#### Kiedy N-gram nadal ma sens
- Małe zasoby obliczeniowe, urządzenia brzegowe, bardzo niska latencja.
- Baseline, autouzupełnianie, korekta pisowni, wykrywanie języka, ASR (wsparcie w rescoringu), filtrowanie/deduplikacja i wykrywanie kontaminacji danych.
- Gdy potrzebna pełna kontrola i wyjaśnialność.

#### Kiedy deep learning
- Zadania wymagające długich zależności, semantyki, generalizacji, generowania (LLM), wielu zadań przez transfer learning.

Neuronowe modele (Bengio et al., 2003) rozwiązały główne wady N-gramów: przekleństwo wymiarowości i brak generalizacji.

**Źródła:**
- [Jurafsky & Martin – Speech and Language Processing, rozdział N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [A Neural Probabilistic Language Model (Bengio et al., JMLR)](https://www.jmlr.org/papers/v3/bengio03a.html)
- [Wikipedia – n-gram](https://en.wikipedia.org/wiki/N-gram)
- [Stanford CS224n – NLP with Deep Learning](https://web.stanford.edu/class/cs224n/)

---

<a id="q132"></a>
### 132. Czym jest model n-gram?

**Odpowiedź:**

**N-gram** to ciągła sekwencja $n$ elementów (słów, znaków, tokenów). Bigram $n=2$, trigram $n=3$. **Model n-gram** to model języka szacujący prawdopodobieństwo następnego słowa na podstawie $n-1$ poprzednich.

#### Podstawy
Z reguły łańcuchowej: $P(w_1..w_T)=\prod_t P(w_t\mid w_{<t})$. Przybliżenie Markowa rzędu $n-1$:
$$P(w_t\mid w_{<t})\approx P(w_t\mid w_{t-n+1},\dots,w_{t-1})$$
Estymacja największej wiarygodności (MLE):
$$P(w_t\mid w_{t-1})=\frac{C(w_{t-1},w_t)}{C(w_{t-1})}$$

#### Przykład
Korpus: „to jest kot", „to jest pies". Bigramy: $C(\text{to jest})=2$, $C(\text{to})=2$, więc $P(\text{jest}\mid\text{to})=1$; $P(\text{kot}\mid\text{jest})=1/2$.

#### Wygładzanie (smoothing)
Nieobserwowane n-gramy dostałyby prawdopodobieństwo 0, co psuje ocenę całych zdań. Rozwiązania:
- **Add-one (Laplace)** / add-k,
- **Backoff / interpolation** (użycie krótszych n-gramów),
- **Kneser–Ney** – najlepsza klasyczna metoda (uwzględnia różnorodność kontekstów słowa),
- praca w logarytmach, by uniknąć underflow.

#### Zastosowania
Rozpoznawanie mowy, autouzupełnianie, korekta pisowni, tłumaczenie statystyczne, klasyfikacja tekstów (n-gramy znakowe – wykrywanie języka), cechy dla klasyfikatorów (TF-IDF z n-gramami).

#### Zalety i wady
- Zalety: prostota, szybkość, interpretowalność, brak potrzeby GPU.
- Wady: rzadkość danych, krótki kontekst, brak generalizacji semantycznej, duży rozmiar tablic dla dużych $n$.

```python
from collections import Counter
tokens = "to jest kot to jest pies".split()
bi = Counter(zip(tokens, tokens[1:])); uni = Counter(tokens)
p = lambda w1, w2: bi[(w1, w2)] / uni[w1]
print(p("to", "jest"))  # 1.0
```

**Źródła:**
- [Jurafsky & Martin – N-gram Language Models (SLP3, rozdział 3)](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [Wikipedia – n-gram](https://en.wikipedia.org/wiki/N-gram)
- [Wikipedia – Kneser–Ney smoothing](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing)

---

<a id="q133"></a>
### 133. Czym jest TF-IDF i czym różni się od word embeddings?

**Odpowiedź:**

**TF-IDF** (Term Frequency – Inverse Document Frequency) to rzadka reprezentacja tekstu, ważąca słowo tym bardziej, im częściej występuje w dokumencie, i tym mniej, im częściej występuje w całym korpusie.

#### Wzory
$$\mathrm{tf\text{-}idf}(t,d)=\mathrm{tf}(t,d)\cdot\mathrm{idf}(t),\qquad \mathrm{idf}(t)=\log\frac{N}{\mathrm{df}(t)}$$
gdzie $N$ – liczba dokumentów, $\mathrm{df}(t)$ – liczba dokumentów zawierających $t$. Warianty: sublinear tf ($1+\log \mathrm{tf}$), wygładzone idf. W scikit-learn: $\mathrm{idf}=\ln\frac{1+N}{1+\mathrm{df}}+1$, po czym wektory są normalizowane L2.

#### Przykład
Słowo „the" występuje w prawie każdym dokumencie -> idf ≈ 0 (mała waga). Słowo „bronchit" w kilku dokumentach medycznych -> duża waga.

#### TF-IDF vs word embeddings
| | TF-IDF | Embeddingi (word2vec, GloVe, BERT) |
|---|---|---|
| Wymiarowość | duża (rozmiar słownika), **rzadkie** wektory | mała (100–1024), **gęste** wektory |
| Uczenie | statystyka zliczeń, brak uczenia | uczone (sieć, faktoryzacja współwystępowania) |
| Semantyka | brak – „auto" i „samochód" niepodobne | podobne słowa mają bliskie wektory |
| Kontekst | brak; kolejność ignorowana | statyczne (word2vec) lub kontekstowe (BERT) |
| Polisemia | nie obsługuje | kontekstowe embeddingi tak |
| Dane | działa na małych zbiorach | wymaga dużych korpusów lub pretrenowanych modeli |
| Interpretowalność | wysoka (każda oś = słowo) | niska |
| Koszt | bardzo niski | wyższy |
| Obsługa OOV | brak dla nowych słów | subwords (fastText, BERT) |

#### Kiedy co
- **TF-IDF**: mocny baseline (klasyfikacja tekstu z regresją logistyczną/SVM), wyszukiwanie leksykalne (BM25 jest jego rozwinięciem), małe dane, potrzeba wyjaśnialności.
- **Embeddingi**: podobieństwo semantyczne, wyszukiwanie semantyczne, RAG, zadania wymagające rozumienia znaczenia. W praktyce często **hybryda** (BM25 + wyszukiwanie wektorowe).

```python
from sklearn.feature_extraction.text import TfidfVectorizer
X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(corpus)
```

**Źródła:**
- [scikit-learn – Text feature extraction (TF-IDF)](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Wikipedia – tf–idf](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Efficient Estimation of Word Representations in Vector Space (word2vec, arXiv)](https://arxiv.org/abs/1301.3781)
- [Jurafsky & Martin – Vector Semantics and Embeddings (SLP3)](https://web.stanford.edu/~jurafsky/slp3/6.pdf)

---

<a id="q134"></a>
### 134. Czym jest Bag-of-Words?

**Odpowiedź:**

**Bag-of-Words (BoW)** to najprostsza reprezentacja tekstu: dokument jest zamieniany na wektor o długości słownika, w którym każda współrzędna to liczba wystąpień (lub obecność 0/1) danego słowa; **kolejność słów jest ignorowana** – stąd „worek".

#### Przykład
Dokumenty: D1 = „kot je rybę", D2 = „pies je kość". Słownik: [kot, je, rybę, pies, kość].
- D1 = [1, 1, 1, 0, 0]
- D2 = [0, 1, 0, 1, 1]

#### Etapy
1. Tokenizacja i normalizacja (małe litery, usunięcie interpunkcji, opcjonalnie stopwords, stemming/lematyzacja).
2. Zbudowanie słownika z korpusu treningowego (ograniczenia: `min_df`, `max_features`).
3. Wektoryzacja (zliczenia, binarnie, lub ważona TF-IDF).

#### Zalety
- Prosta, szybka, interpretowalna; dobra baza dla klasyfikacji tekstu (Naive Bayes, regresja logistyczna, SVM), filtrów spamu i wykrywania tematów.
- Działa z małymi danymi.

#### Wady
- **Utrata kolejności i kontekstu** („pies gryzie człowieka" = „człowiek gryzie psa"); częściowo łagodzą to n-gramy.
- **Wysoka wymiarowość i rzadkość** (słownik 50k–1M).
- **Brak semantyki** – synonimy niezwiązane; nie obsługuje nowych słów (OOV) spoza słownika.
- Częste słowa dominują (rozwiązanie: TF-IDF, stopwords).
- Wyciek danych, jeśli słownik budowany jest na całości danych – trzeba dopasować (`fit`) tylko na zbiorze treningowym.

#### Alternatywy
TF-IDF, hashing trick (`HashingVectorizer`), embeddingi (word2vec, fastText), modele kontekstowe (BERT), uśrednianie embeddingów.

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(["kot je rybę", "pies je kość"])
print(cv.get_feature_names_out(), X.toarray())
```

**Źródła:**
- [scikit-learn – Text feature extraction (bag of words)](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Wikipedia – Bag-of-words model](https://en.wikipedia.org/wiki/Bag-of-words_model)
- [scikit-learn – CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

---

<a id="q135"></a>
### 135. Do czego służy perplexity w NLP?

**Odpowiedź:**

**Perplexity (PPL)** to standardowa wewnętrzna (intrinsic) metryka jakości **modelu języka**: mierzy, jak „zaskoczony" jest model tekstem testowym. Im niższa, tym lepiej model przewiduje dane.

#### Definicja
Dla sekwencji $w_1..w_N$:
$$\mathrm{PPL}=P(w_1..w_N)^{-1/N}=\exp\!\Big(-\frac1N\sum_{i=1}^N\log P(w_i\mid w_{<i})\Big)=\exp(H)$$
czyli wykładnik średniej cross-entropy (w nat; przy log₂ używa się $2^H$).

#### Interpretacja
- PPL = $k$ oznacza, że model jest średnio tak niepewny, jakby wybierał jednostajnie spośród $k$ równie prawdopodobnych tokenów (efektywny współczynnik rozgałęzienia).
- PPL = 1 – idealne przewidywanie; losowy model z słownikiem $|V|$ ma PPL = $|V|$.
- Przykład: jeśli model przypisuje każdemu tokenowi 0,1, to PPL = 10.

#### Zastosowania
- Porównywanie modeli językowych (N-gram vs LSTM vs Transformer) na tym samym zbiorze.
- Monitorowanie treningu i strojenie hiperparametrów, wykrywanie overfittingu (PPL walidacyjna).
- Filtrowanie danych (odrzucanie tekstu o zbyt wysokiej PPL względem małego modelu), wykrywanie treści generowanych przez AI, ocena adaptacji domenowej.

#### Ograniczenia
- **Porównywalna tylko przy tej samej tokenizacji i słowniku** oraz tym samym zbiorze testowym (PPL na poziomie subwordów vs słów vs znaków nie jest porównywalna; stosuje się bits-per-character/byte).
- Niska PPL nie gwarantuje przydatności w zadaniu (faktografia, bezpieczeństwo, zgodność z instrukcją), dlatego oprócz niej stosuje się ewaluacje zadaniowe (benchmarki, human eval, LLM-as-judge).
- Nie nadaje się do modeli maskowanych bez modyfikacji (pseudo-perplexity dla BERT).
- Wrażliwa na wyciek danych testowych do treningu.

```python
import torch, math
loss = model(**batch, labels=batch["input_ids"]).loss  # średnia cross-entropy
ppl = math.exp(loss.item())
```

**Źródła:**
- [Hugging Face – Perplexity of fixed-length models](https://huggingface.co/docs/transformers/perplexity)
- [Jurafsky & Martin – N-gram Language Models (perplexity)](https://web.stanford.edu/~jurafsky/slp3/3.pdf)
- [Wikipedia – Perplexity](https://en.wikipedia.org/wiki/Perplexity)

---

<a id="q136"></a>
### 136. Czym różni się stemming od lematyzacji?

**Odpowiedź:**

Obie techniki normalizują słowa do formy bazowej, aby zmniejszyć liczbę wariantów w słowniku (np. w wyszukiwaniu i klasyfikacji tekstu), ale robią to inaczej.

#### Stemming
- **Heurystyczne obcinanie końcówek** (reguły) bez znajomości słownika i kontekstu; wynik (stem) nie musi być poprawnym słowem.
- Algorytmy: Porter, Snowball (Porter2), Lancaster.
- Przykłady (angielski): *running, runs -> run*; *studies -> studi*; *better -> better*; *universal, university, universe* bywają sprowadzane do wspólnego rdzenia (overstemming); *data, datum* mogą się nie połączyć (understemming).
- Zalety: bardzo szybki, prosty, niezależny od słownika. Wady: błędy nad- i niedostatecznego skracania; słabo działa w językach fleksyjnych, takich jak polski (bogata morfologia).

#### Lematyzacja
- Sprowadzenie słowa do **lematu** (formy słownikowej) z użyciem słownika/morfologii i często części mowy (POS) i kontekstu.
- Przykłady: *better -> good* (przy POS = przymiotnik), *was -> be*, *studies -> study*; po polsku: *psami, psa, psy -> pies*, *poszedłem -> pójść*.
- Zalety: poprawne, interpretowalne formy; lepsza jakość semantyczna. Wady: wolniejsza, wymaga zasobów językowych (WordNet, spaCy, Morfeusz, Stanza).

| | Stemming | Lematyzacja |
|---|---|---|
| Metoda | reguły obcinania | słownik + morfologia + POS |
| Wynik | rdzeń (może nie być słowem) | poprawny lemat |
| Szybkość | wysoka | niższa |
| Dokładność | niższa | wyższa |
| Języki fleksyjne (PL) | słabo | lepiej (polecane) |

#### Kiedy co
- **Stemming**: wyszukiwanie/indeksowanie w dużej skali, gdy liczy się szybkość i przypomnienie (recall).
- **Lematyzacja**: gdy ważna jest precyzja i czytelność (analiza tekstu, klasyfikacja, tematyka), szczególnie dla polskiego.
- W nowoczesnych modelach opartych na subwordach (BERT, LLM) ani jedno, ani drugie zwykle nie jest potrzebne – tokenizer subwordowy i kontekstowe embeddingi radzą sobie z fleksją.

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
print(PorterStemmer().stem("studies"))            # studi
print(WordNetLemmatizer().lemmatize("studies"))   # study
```

**Źródła:**
- [Stanford IR Book – Stemming and lemmatization](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)
- [NLTK – nltk.stem package](https://www.nltk.org/api/nltk.stem.html)
- [Wikipedia – Stemming](https://en.wikipedia.org/wiki/Stemming)
- [Wikipedia – Lemmatization](https://en.wikipedia.org/wiki/Lemmatisation)

---

<a id="q137"></a>
### 137. Czym jest Latent Semantic Indexing (LSI)?

**Odpowiedź:**

**Latent Semantic Indexing (LSI)**, zwane też Latent Semantic Analysis (LSA), to klasyczna metoda redukcji wymiarowości macierzy term-dokument oparta na rozkładzie SVD (Singular Value Decomposition). Jej celem jest wykrycie ukrytych ("latentnych") struktur semantycznych: słowa, które współwystępują w podobnych dokumentach, lądują blisko siebie w przestrzeni niskowymiarowej, nawet jeśli nigdy nie pojawiły się w tym samym dokumencie.

#### Mechanizm

1. Budujemy macierz $A \in \mathbb{R}^{m \times n}$ (m termów, n dokumentów), zwykle z wagami TF-IDF.
2. Wykonujemy SVD: $A = U \Sigma V^\top$.
3. Zachowujemy tylko $k$ największych wartości osobliwych (zwykle rzędu 100-300):
$$A_k = U_k \Sigma_k V_k^\top$$
Według twierdzenia Eckarta-Younga $A_k$ jest najlepszą aproksymacją rzędu $k$ w normie Frobeniusa.
4. Dokumenty są reprezentowane przez wiersze $V_k \Sigma_k$, termy przez $U_k \Sigma_k$. Zapytanie $q$ rzutujemy: $\hat q = \Sigma_k^{-1} U_k^\top q$, a podobieństwo liczymy kosinusowo.

#### Co to daje

- **Synonimia**: "auto" i "samochód" mają podobne wektory, więc zapytanie o jedno znajduje dokumenty z drugim.
- **Polisemia**: tylko częściowo rozwiązana (jedno słowo = jeden wektor).
- Redukcja szumu i wymiaru, gęste reprezentacje.

#### Ograniczenia

- Liniowy model, nie uwzględnia kolejności słów (bag-of-words).
- Składowe mogą mieć wartości ujemne i są trudne do interpretacji (w przeciwieństwie do NMF czy LDA).
- SVD jest kosztowne dla bardzo dużych korpusów (stosuje się randomized SVD).
- Słabsze niż nowoczesne embeddingi kontekstowe.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

lsa = make_pipeline(TfidfVectorizer(stop_words="english"),
                    TruncatedSVD(n_components=100, random_state=0))
X_lsa = lsa.fit_transform(docs)   # (n_docs, 100)
```

**Źródła:**
- [Wikipedia: Latent semantic analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
- [scikit-learn: TruncatedSVD (LSA)](https://scikit-learn.org/stable/modules/decomposition.html#lsa)
- [Stanford IR Book: Latent semantic indexing](https://nlp.stanford.edu/IR-book/html/htmledition/latent-semantic-indexing-1.html)

---

<a id="q138"></a>
### 138. Czym jest dependency parsing (analiza zależnościowa)?

**Odpowiedź:**

**Dependency parsing** to zadanie NLP polegające na wyznaczeniu struktury składniowej zdania w postaci **drzewa zależności**: każde słowo (poza korzeniem, zwykle głównym czasownikiem) ma dokładnie jednego "nadrzędnika" (head), a krawędź jest opisana relacją gramatyczną (np. `nsubj`, `obj`, `amod`, `obl`). W przeciwieństwie do constituency parsing (drzewa fraz) nie tworzy węzłów pośrednich, tylko relacje słowo-słowo.

#### Przykład

"Ala ma kota" → `ma` (root), `Ala` --nsubj--> `ma`, `kota` --obj--> `ma`.

#### Podejścia

- **Transition-based** (shift-reduce, np. arc-standard): parser czyta zdanie od lewej do prawej i podejmuje decyzje SHIFT / LEFT-ARC / RIGHT-ARC klasyfikatorem (dziś sieć neuronowa). Złożoność liniowa, bardzo szybkie (spaCy).
- **Graph-based**: ocenia wszystkie możliwe krawędzie (biaffine attention) i szuka maksymalnego drzewa rozpinającego (algorytm Chu-Liu/Edmonds). Zwykle nieco dokładniejsze, koszt $O(n^2)$–$O(n^3)$.
- **Neuronowe**: BiLSTM lub Transformer jako enkoder + biaffine klasyfikator (Dozat & Manning); obecnie standardem są modele oparte na BERT/XLM-R.

#### Własności i pojęcia

- **Projektywność**: drzewo bez skrzyżowanych krawędzi; języki o swobodnym szyku (jak polski) mają więcej struktur nieprojektywnych.
- Metryki: **UAS** (odsetek poprawnych headów) i **LAS** (poprawny head + etykieta).
- Standard adnotacji: **Universal Dependencies** (spójny zestaw relacji dla >100 języków).

#### Zastosowania

Ekstrakcja informacji (relacje podmiot-orzeczenie-dopełnienie), analiza sentymentu aspektowego, tłumaczenie maszynowe, odpowiadanie na pytania, semantic role labeling, sprawdzanie gramatyki.

```python
import spacy
nlp = spacy.load("en_core_web_sm")
for t in nlp("The cat chased the mouse"):
    print(t.text, t.dep_, t.head.text)
```

**Źródła:**
- [Universal Dependencies](https://universaldependencies.org/)
- [Jurafsky & Martin, Speech and Language Processing (rozdział Dependency Parsing)](https://web.stanford.edu/~jurafsky/slp3/)
- [Wikipedia: Dependency grammar](https://en.wikipedia.org/wiki/Dependency_grammar)
- [spaCy: Linguistic features](https://spacy.io/usage/linguistic-features)

---

<a id="q139"></a>
### 139. Jakie są podejścia do streszczania tekstu (text summarization)?

**Odpowiedź:**

Streszczanie tekstu dzieli się na dwie główne rodziny oraz warianty pośrednie.

#### 1. Ekstrakcyjne (extractive)

Wybieramy z oryginału najważniejsze zdania i sklejamy je bez zmian.

- **TextRank / LexRank**: graf zdań z krawędziami wg podobieństwa, ranking przez PageRank.
- **TF-IDF / częstość słów, pozycja zdania, cue words**.
- **LSA** – wybór zdań wg dominujących składowych SVD.
- **Neuronowe**: klasyfikacja każdego zdania (BERTSum) – czy włączyć do streszczenia.
- Zalety: wierność faktom, gramatyczność, prostota. Wady: brak spójności, redundancja, brak parafrazy.

#### 2. Abstrakcyjne (abstractive)

Model generuje nowy tekst, podobnie jak człowiek.

- **Seq2seq z attention** (RNN), **pointer-generator** (kopiowanie słów ze źródła + generowanie).
- **Transformery enkoder-dekoder**: BART, T5, PEGASUS (pre-training z maskowaniem całych zdań "gap sentences").
- **Duże modele językowe (LLM)** z promptem lub fine-tuningiem; długie dokumenty przez chunking, map-reduce lub refine.
- Zalety: płynność, kompresja, parafraza. Wady: **halucynacje** (niewierność źródłu), koszt.

#### 3. Hybrydowe

Najpierw selekcja kluczowych fragmentów (ekstrakcja), potem przepisanie modelem abstrakcyjnym; użyteczne dla dokumentów przekraczających okno kontekstu.

#### Ewaluacja

- **ROUGE-1/2/L** (nakładanie n-gramów z referencją), **BLEU**, **METEOR**.
- **BERTScore** (podobieństwo embeddingów), metryki faktograficzne (factual consistency, np. oparte o NLI/QA) oraz ocena przez LLM/ludzi. ROUGE słabo mierzy prawdziwość.

#### Wskazówki praktyczne

Dla dokumentów prawnych/medycznych rozważ ekstrakcję lub abstrakcję z cytowaniem źródeł. Kontroluj długość (`max_length`, `length_penalty`), użyj beam search, a dla długich tekstów hierarchicznego streszczania.

```python
from transformers import pipeline
summ = pipeline("summarization", model="facebook/bart-large-cnn")
print(summ(text, max_length=120, min_length=40)[0]["summary_text"])
```

**Źródła:**
- [TextRank: Bringing Order into Texts (Mihalcea & Tarau)](https://aclanthology.org/W04-3252/)
- [PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization](https://arxiv.org/abs/1912.08777)
- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- [Hugging Face: Summarization task guide](https://huggingface.co/docs/transformers/tasks/summarization)

---

<a id="q140"></a>
### 140. Czym są word embeddings (zanurzenia słów)?

**Odpowiedź:**

**Word embeddings** to gęste, niskowymiarowe (zwykle 50-1024) wektory liczb rzeczywistych reprezentujące słowa tak, że **podobieństwo semantyczne odpowiada bliskości geometrycznej**. Zastępują rzadkie reprezentacje one-hot / bag-of-words, w których każde słowo jest ortogonalne do wszystkich innych i wektor ma rozmiar słownika.

#### Idea: hipoteza dystrybucyjna

"Słowo poznaje się po jego otoczeniu" (Firth). Embeddingi uczy się z kontekstów występowania słów w dużych korpusach.

#### Główne metody

| Metoda | Idea |
|---|---|
| **Word2Vec** (CBOW, Skip-gram) | predykcja słowa z kontekstu lub kontekstu ze słowa, płytka sieć |
| **GloVe** | faktoryzacja logarytmu globalnej macierzy współwystępowania |
| **fastText** | wektor słowa = suma wektorów n-gramów znakowych, radzi sobie z OOV i morfologią (ważne dla polskiego) |
| **ELMo / BERT / GPT** | embeddingi **kontekstowe**: wektor zależy od zdania ("zamek" - budowla vs. zamek błyskawiczny) |

#### Własności

- Struktura liniowa: $\vec{król} - \vec{mężczyzna} + \vec{kobieta} \approx \vec{królowa}$ (analogie).
- Podobieństwo mierzymy kosinusem: $\cos(u,v) = \frac{u \cdot v}{\|u\|\|v\|}$.
- Embeddingi statyczne dają jeden wektor na słowo (problem polisemii); kontekstowe to rozwiązują.

#### Praktyka i pułapki

- Można użyć wag wstępnie wytrenowanych (transfer learning) lub uczyć warstwę `nn.Embedding` razem z zadaniem.
- Embeddingi przejmują **uprzedzenia** z danych (np. stereotypy płciowe).
- Zdania/dokumenty: uśrednianie wektorów jest prostym baseline, lepsze są modele typu Sentence-BERT.

```python
import torch.nn as nn
emb = nn.Embedding(num_embeddings=50_000, embedding_dim=300, padding_idx=0)
```

**Źródła:**
- [Efficient Estimation of Word Representations in Vector Space (Word2Vec)](https://arxiv.org/abs/1301.3781)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Enriching Word Vectors with Subword Information (fastText)](https://arxiv.org/abs/1607.04606)
- [Wikipedia: Word embedding](https://en.wikipedia.org/wiki/Word_embedding)

---

<a id="q141"></a>
### 141. Czym jest Word2Vec?

**Odpowiedź:**

**Word2Vec** (Mikolov i in., Google, 2013) to rodzina prostych, szybkich modeli uczących **statyczne embeddingi słów** z surowego tekstu, bez etykiet (uczenie samonadzorowane). Sieć jest płytka: warstwa wejściowa (one-hot) → warstwa ukryta liniowa (macierz embeddingów $W$) → warstwa wyjściowa (softmax po słowniku). Po treningu używa się macierzy $W$ jako wektorów słów.

#### Dwie architektury

- **CBOW (Continuous Bag of Words)**: przewiduje słowo centralne na podstawie uśrednionych wektorów słów z okna kontekstu. Szybszy, lepszy dla częstych słów.
- **Skip-gram**: przewiduje słowa kontekstu na podstawie słowa centralnego. Wolniejszy, ale lepszy dla rzadkich słów i mniejszych korpusów.

#### Funkcja celu

Skip-gram maksymalizuje
$$\frac{1}{T}\sum_{t}\sum_{-c\le j\le c,\, j\ne0}\log p(w_{t+j}\mid w_t),\quad p(o\mid c)=\frac{\exp(u_o^\top v_c)}{\sum_{w\in V}\exp(u_w^\top v_c)}$$
Pełny softmax kosztuje $O(|V|)$, dlatego stosuje się przybliżenia:

- **Negative sampling**: dla każdej pozytywnej pary losujemy $k$ (zwykle 5-20) negatywnych słów i uczymy klasyfikator binarny: $\log\sigma(u_o^\top v_c)+\sum_{i=1}^k \mathbb{E}\log\sigma(-u_{w_i}^\top v_c)$.
- **Hierarchical softmax**: drzewo Huffmana, koszt $O(\log|V|)$.
- Dodatkowo **subsampling** częstych słów i rozkład negatywów $\propto f(w)^{3/4}$.

#### Hiperparametry

Wymiar (100-300), rozmiar okna (5-10), `min_count`, liczba negatywów, epoki.

#### Ograniczenia

Jeden wektor na słowo (brak polisemii), brak obsługi słów spoza słownika (fastText to naprawia), nie uwzględnia kontekstu zdania. Mimo to bywa użyteczny jako lekki baseline, np. w rekomendacjach (item2vec).

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences, vector_size=200, window=5, sg=1, negative=10, min_count=5)
model.wv.most_similar("king")
```

**Źródła:**
- [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
- [Gensim: Word2Vec documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [Stanford CS224n: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)

---

<a id="q142"></a>
### 142. Czym jest t-SNE i jak jest używane w NLP?

**Odpowiedź:**

**t-SNE (t-distributed Stochastic Neighbor Embedding)** to nieliniowa metoda redukcji wymiarowości służąca głównie do **wizualizacji** danych wysokowymiarowych w 2D/3D, zachowująca strukturę lokalną (sąsiedztwo).

#### Mechanizm

1. W przestrzeni oryginalnej dla punktów $x_i$ definiujemy rozkład podobieństw (gaussowski, szerokość dobierana przez **perplexity**):
$$p_{j|i}=\frac{\exp(-\|x_i-x_j\|^2/2\sigma_i^2)}{\sum_{k\ne i}\exp(-\|x_i-x_k\|^2/2\sigma_i^2)},\quad p_{ij}=\frac{p_{j|i}+p_{i|j}}{2n}$$
2. W przestrzeni 2D używamy rozkładu t-Studenta z 1 stopniem swobody (ciężkie ogony, redukuje "crowding problem"):
$$q_{ij}=\frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\ne l}(1+\|y_k-y_l\|^2)^{-1}}$$
3. Minimalizujemy dywergencję KL $\mathrm{KL}(P\|Q)$ metodą gradientu.

#### Zastosowania w NLP

- Wizualizacja **word embeddings** (Word2Vec, GloVe): klastry tematyczne, analogie.
- Eksploracja embeddingów zdań/dokumentów, klastrów tematów, wyjść warstw BERT.
- Debugowanie: wykrywanie błędnych etykiet, wpływu bias, separacji klas w przestrzeni cech.

#### Pułapki interpretacji

- **Odległości między klastrami, ich rozmiary i gęstość nie mają znaczenia globalnego**; liczy się tylko lokalne sąsiedztwo.
- Wynik zależy od perplexity (zwykle 5-50), inicjalizacji i seedu; warto próbować kilku wartości.
- Nie służy do budowania cech dla modeli (brak transformacji nowych punktów); do tego PCA/UMAP lub autoenkoder.
- Złożoność $O(n^2)$; wariant Barnes-Hut $O(n\log n)$. Przy dużych wektorach najpierw PCA do ~50 wymiarów.

```python
from sklearn.manifold import TSNE
Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(X)
```

**Źródła:**
- [Visualizing Data using t-SNE (van der Maaten & Hinton, JMLR 2008)](https://jmlr.org/papers/v9/vandermaaten08a.html)
- [How to Use t-SNE Effectively (Distill)](https://distill.pub/2016/misread-tsne/)
- [scikit-learn: TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

---

<a id="q143"></a>
### 143. Wyjaśnij ColBERT

**Odpowiedź:**

**ColBERT (Contextualized Late Interaction over BERT)** to architektura wyszukiwania (retrieval), która łączy jakość cross-enkoderów z szybkością bi-enkoderów dzięki **późnej interakcji (late interaction)** na poziomie tokenów.

#### Porównanie podejść

- **Bi-encoder (dense retrieval, np. DPR)**: cały dokument i zapytanie kompresowane do jednego wektora; szybkie, ale traci szczegóły.
- **Cross-encoder**: zapytanie i dokument razem przez BERT; dokładne, ale trzeba przeliczyć każdą parę, więc niepraktyczne dla milionów dokumentów.
- **ColBERT**: osobno koduje zapytanie i dokument do **zbioru wektorów tokenów** (po jednym na token), a interakcję liczy dopiero na końcu, tanią operacją.

#### Mechanizm: MaxSim

Niech $E_q$ i $E_d$ to znormalizowane embeddingi tokenów zapytania i dokumentu. Wynik:
$$S(q,d)=\sum_{i\in E_q}\max_{j\in E_d}\; E_{q_i}\cdot E_{d_j}$$
Dla każdego tokenu zapytania szukamy najbardziej podobnego tokenu dokumentu i sumujemy. Daje to dopasowanie "miękkie", zbliżone do BM25 na poziomie słów, ale semantyczne i kontekstowe.

#### Zalety

- Embeddingi dokumentów można **policzyć offline i zaindeksować** (FAISS/ANN); w czasie zapytania koduje się tylko zapytanie.
- Znacznie wyższa jakość niż pojedynczy wektor, koszt o rzędy wielkości niższy niż cross-encoder.
- Interpretowalność (widać, które tokeny się dopasowały).

#### Wady i rozwinięcia

- Duży indeks (wektor na token); **ColBERTv2** stosuje kompresję resztkową i supervision typu denoising, redukując ślad pamięci; **PLAID** przyspiesza wyszukiwanie.
- Wersje wielojęzyczne i multimodalne (ColPali dla dokumentów-obrazów).
- Zwykle stosowany jako retriever lub reranker w pipeline RAG.

**Źródła:**
- [Decoding ColBERT (Outcome School)](https://outcomeschool.com/blog/decoding-colbert)
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)
- [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)

---

## Computer Vision (widzenie komputerowe)

<a id="q144"></a>
### 144. Czym jest computer vision i dlaczego jest ważne?

**Odpowiedź:**

**Computer vision (CV)** to dziedzina AI, której celem jest umożliwienie maszynom "rozumienia" obrazów i wideo: wydobywania z pikseli informacji o obiektach, scenie, ruchu i geometrii 3D. To trudne, bo obraz to tylko macierz liczb, a te same obiekty wyglądają inaczej przy zmianie oświetlenia, kąta widzenia, skali, okluzji czy tła (tzw. semantic gap).

#### Główne zadania

- **Klasyfikacja obrazów** (co jest na zdjęciu),
- **Detekcja obiektów** (co i gdzie: bounding boxy),
- **Segmentacja** (semantyczna, instancji, panoptyczna),
- **Estymacja pozy, śledzenie, rozpoznawanie twarzy**,
- **OCR**, estymacja głębi, rekonstrukcja 3D, optical flow,
- **Generowanie i opisywanie obrazów** (diffusion, image captioning, VQA).

#### Dlaczego jest ważne

- **Medycyna**: analiza RTG, MRI, patologii, wykrywanie zmian nowotworowych.
- **Motoryzacja**: autonomiczna jazda, ADAS.
- **Przemysł**: kontrola jakości, wykrywanie defektów, robotyka.
- **Handel i bezpieczeństwo**: sklepy bezobsługowe, monitoring, biometria.
- **Rolnictwo i środowisko**: monitoring upraw, zdjęcia satelitarne.
- **Dostępność** i codzienne aplikacje (wyszukiwanie zdjęć, AR, skanowanie dokumentów).

#### Ewolucja

Klasyczne metody (SIFT, HOG + SVM, ręcznie projektowane cechy) zostały wyparte przez **CNN** (AlexNet 2012, ResNet), a obecnie coraz częściej przez **Vision Transformers** i modele fundamentowe (CLIP, SAM, DINOv2).

#### Wyzwania

Zmienność danych (domain shift), potrzeba dużych zbiorów z etykietami, wydajność w czasie rzeczywistym na urządzeniach brzegowych, odporność na ataki adversarial, kwestie prywatności i uprzedzeń (bias) w rozpoznawaniu twarzy.

**Źródła:**
- [Stanford CS231n: Deep Learning for Computer Vision](http://cs231n.stanford.edu/)
- [Szeliski: Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)
- [Wikipedia: Computer vision](https://en.wikipedia.org/wiki/Computer_vision)

---

<a id="q145"></a>
### 145. Czym jest segmentacja obrazu i jakie są jej zastosowania?

**Odpowiedź:**

**Segmentacja obrazu** to przypisanie etykiety każdemu pikselowi, czyli klasyfikacja na poziomie pikseli. Daje znacznie dokładniejszą lokalizację niż bounding boxy.

#### Rodzaje

- **Semantic segmentation**: każdy piksel dostaje klasę (droga, niebo, człowiek); nie rozróżnia instancji tej samej klasy.
- **Instance segmentation**: oddzielna maska dla każdego obiektu (Mask R-CNN).
- **Panoptic segmentation**: połączenie obu (klasy "stuff" jak niebo + instancje "things" jak samochody).
- Segmentacja interaktywna / promptowalna (Segment Anything, SAM).

#### Architektury

- **FCN**: sieć w pełni konwolucyjna, zastępuje warstwy FC konwolucjami 1×1, wyjście to mapa klas po upsamplingu.
- **U-Net**: enkoder-dekoder ze **skip connections** łączącymi warstwy o tej samej rozdzielczości; standard w obrazowaniu medycznym, dobrze działa na małych zbiorach.
- **DeepLab**: konwolucje atrous (dylatowane), ASPP dla wielu skal.
- **Mask R-CNN**: dodaje gałąź maski do Faster R-CNN.
- **Transformery**: SegFormer, Mask2Former, SAM.

#### Funkcje straty i metryki

- Cross-entropy (często ważona przy niezbalansowanych klasach), **Dice loss** $=1-\frac{2|A\cap B|}{|A|+|B|}$, focal loss.
- **IoU / mIoU**, Dice coefficient, pixel accuracy (myląca przy dominującym tle).

#### Zastosowania

Obrazowanie medyczne (organy, guzy), jazda autonomiczna (droga, pasy), zdjęcia satelitarne i rolnictwo, robotyka (chwytanie), edycja zdjęć (usuwanie tła, portret), inspekcja przemysłowa, AR.

```python
import torch, torchvision
model = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT").eval()
out = model(torch.randn(1, 3, 512, 512))["out"]   # (1, 21, 512, 512)
mask = out.argmax(1)
```

**Źródła:**
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [Mask R-CNN](https://arxiv.org/abs/1703.06870)
- [Segment Anything](https://arxiv.org/abs/2304.02643)

---

<a id="q146"></a>
### 146. Czym jest detekcja obiektów i czym różni się od klasyfikacji obrazów?

**Odpowiedź:**

**Klasyfikacja obrazu** odpowiada na pytanie "co jest na obrazie?" i zwraca jedną etykietę (lub wektor prawdopodobieństw) dla całego obrazu. **Detekcja obiektów** odpowiada na "co i gdzie?": dla każdego obiektu zwraca **bounding box** (współrzędne), **klasę** i **score** pewności; liczba obiektów jest zmienna i nieznana z góry.

| | Klasyfikacja | Detekcja |
|---|---|---|
| Wyjście | klasa | zestaw (box, klasa, score) |
| Loss | cross-entropy | klasyfikacja + regresja boxów (+ objectness) |
| Metryka | accuracy, F1 | **mAP** przy progach IoU |
| Trudność | jeden obiekt dominujący | wiele obiektów, skale, okluzje |

#### Rodziny detektorów

- **Dwuetapowe** (R-CNN, Fast/Faster R-CNN): najpierw propozycje regionów (RPN), potem klasyfikacja i dopracowanie boxów; dokładne, wolniejsze.
- **Jednoetapowe** (YOLO, SSD, RetinaNet): bezpośrednia predykcja boxów i klas z mapy cech na siatce/anchorach; szybkie, dobre do czasu rzeczywistego. RetinaNet wprowadził **focal loss** przeciw niezbalansowaniu tło/obiekt.
- **Transformerowe** (DETR): traktuje detekcję jako predykcję zbioru, dopasowanie Hungarian matching, bez anchorów i NMS.

#### Kluczowe pojęcia

- **IoU** (Intersection over Union) = pole przecięcia / pole sumy boxów.
- **NMS (Non-Maximum Suppression)**: usuwa nakładające się duplikaty, zostawiając box o najwyższym score.
- **Anchor boxes** (zdefiniowane proporcje) vs. anchor-free (FCOS, CenterNet).
- **mAP**: średnia z AP po klasach (COCO: uśrednienie po IoU 0.5:0.95).

```python
import torchvision
m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT").eval()
pred = m([img_tensor])[0]   # dict: boxes, labels, scores
```

**Źródła:**
- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [End-to-End Object Detection with Transformers (DETR)](https://arxiv.org/abs/2005.12872)
- [Focal Loss for Dense Object Detection (RetinaNet)](https://arxiv.org/abs/1708.02002)

---

<a id="q147"></a>
### 147. Jakie są kroki budowy systemu rozpoznawania obrazów?

**Odpowiedź:**

#### 1. Zdefiniuj problem i metryki

Klasyfikacja / detekcja / segmentacja? Jakie klasy, jakie ograniczenia (latencja, urządzenie brzegowe, koszt błędów FP vs FN)? Wybierz metrykę zgodną z biznesem (F1, recall, mAP) i ustal baseline.

#### 2. Dane

- Zbieranie (własne, publiczne, syntetyczne), **etykietowanie** (jasne wytyczne, kontrola jakości, zgodność między annotatorami, narzędzia jak CVAT, Label Studio).
- Podział train/val/test **bez wycieków** (np. zdjęcia tego samego pacjenta/obiektu w jednym zbiorze).
- Analiza rozkładu klas, jakości, błędnych etykiet; balansowanie klas.

#### 3. Preprocessing i augmentacja

Zmiana rozmiaru, normalizacja (średnia/odchylenie ImageNet), augmentacje (flip, crop, color jitter, mixup) dopasowane do domeny.

#### 4. Model

Zacznij od **transfer learning**: wytrenowany backbone (ResNet, EfficientNet, ConvNeXt, ViT) + nowa głowica. Najpierw trenuj głowicę, potem odblokuj wyższe warstwy z małym learning rate. Dla dużych zbiorów trenuj od zera lub fine-tuninguj w całości.

#### 5. Trening

Optymalizator (AdamW/SGD+momentum), scheduler (cosine), mixed precision, early stopping, regularyzacja (weight decay, dropout, label smoothing), śledzenie eksperymentów (MLflow, W&B).

#### 6. Ewaluacja i analiza błędów

Macierz pomyłek, metryki per-klasa, analiza przypadków granicznych, testy odporności (oświetlenie, rozmycie, inne kamery), kalibracja pewności.

#### 7. Optymalizacja i wdrożenie

Kwantyzacja, pruning, distillation, eksport ONNX/TensorRT, serwowanie (batch, GPU/edge), monitoring **data drift**, pętla ponownego trenowania z trudnymi przypadkami (active learning).

**Źródła:**
- [Stanford CS231n: Transfer Learning](http://cs231n.github.io/transfer-learning/)
- [PyTorch: Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Google: Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [torchvision documentation](https://pytorch.org/vision/stable/index.html)

---

<a id="q148"></a>
### 148. Jakie są wyzwania w śledzeniu obiektów w czasie rzeczywistym (real-time object tracking)?

**Odpowiedź:**

**Object tracking** to utrzymanie tożsamości (ID) obiektów przez kolejne klatki wideo. Dominującym paradygmatem jest **tracking-by-detection**: detektor (np. YOLO) na każdej klatce, następnie **asocjacja** detekcji z istniejącymi torami (tracks).

#### Główne wyzwania

- **Okluzje**: obiekt zasłonięty częściowo lub całkowicie; ryzyko utraty toru lub zamiany ID (**ID switch**).
- **Podobny wygląd**: tłum, osoby w tych samych ubraniach, pojazdy tej samej marki.
- **Zmiana wyglądu**: skala, poza, oświetlenie, rozmycie ruchu, deformacje.
- **Nieregularny ruch** i szybkie zmiany kierunku; ruch kamery.
- **Wejścia/wyjścia z kadru**, nowe obiekty, fałszywe detekcje i braki detekcji (missed detections).
- **Budżet czasowy**: real-time (np. 25-30 FPS) wymaga zmieszczenia detekcji, ekstrakcji cech i asocjacji w ~30-40 ms; ograniczenia sprzętu edge.
- **Wielu obiektów naraz** (MOT): złożoność asocjacji rośnie.
- **Dryf** przy trackerach opartych o szablon (aktualizacja wyglądu wprowadza błędy).
- **Multi-camera** i re-identification między kamerami.

#### Typowe rozwiązania

- **Filtr Kalmana** do predykcji pozycji + **algorytm węgierski (Hungarian)** do dopasowania po IoU (**SORT**).
- **DeepSORT**: dodaje embedding wyglądu (Re-ID) do kosztu dopasowania, lepiej radzi sobie z okluzjami.
- **ByteTrack**: wykorzystuje także detekcje o niskim score przy asocjacji, ograniczając utraty toru.
- Kompensacja ruchu kamery, buforowanie torów "utraconych" przez kilka klatek.
- Optymalizacje: mniejszy detektor, kwantyzacja, TensorRT, detekcja co N klatek + tracker lekki (KCF, optical flow) pomiędzy.

#### Metryki

**MOTA**, **IDF1**, **HOTA**, liczba ID switchy, FPS.

**Źródła:**
- [Simple Online and Realtime Tracking (SORT)](https://arxiv.org/abs/1602.00763)
- [Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT)](https://arxiv.org/abs/1703.07402)
- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.06864)

---

<a id="q149"></a>
### 149. Czym jest ekstrakcja cech (feature extraction) w computer vision?

**Odpowiedź:**

**Feature extraction** to przekształcenie surowych pikseli w bardziej zwarte, informatywne i niezmiennicze (invariant) reprezentacje (cechy), na których łatwiej działa klasyfikator lub inny algorytm.

#### Klasyczne (ręcznie projektowane) cechy

- **Krawędzie i narożniki**: Sobel, Canny, Harris, Shi-Tomasi.
- **SIFT / SURF**: lokalne deskryptory niezmiennicze względem skali i rotacji; **ORB** - szybka, darmowa alternatywa (binarny deskryptor) używana w SLAM.
- **HOG (Histogram of Oriented Gradients)**: histogramy kierunków gradientów w komórkach; klasyczny detektor pieszych z SVM.
- **LBP**, cechy koloru (histogramy), tekstury (filtry Gabora).
- **Bag of Visual Words**: klastrowanie deskryptorów (k-means) i histogram słów.

Zastosowania: dopasowywanie obrazów, panoramy, rekonstrukcja 3D, śledzenie punktów.

#### Cechy uczone (deep learning)

Sieci CNN uczą hierarchii cech: pierwsze warstwy wykrywają krawędzie i tekstury, środkowe wzory i części, głębokie obiekty i pojęcia. **Backbone** wytrenowany na ImageNet daje uniwersalne cechy do transfer learningu; można użyć wyjścia przed warstwą klasyfikacyjną jako embeddingu obrazu (wyszukiwanie podobnych obrazów, klastrowanie). Modele samonadzorowane (DINO, CLIP) dają cechy bardzo ogólne.

#### Porównanie

| | Ręczne | Uczone |
|---|---|---|
| Dane | mało | dużo |
| Interpretowalność | wysoka | niska |
| Jakość | ograniczona | zwykle znacznie wyższa |
| Koszt obliczeń | niski | wyższy |

```python
import cv2
orb = cv2.ORB_create(1000)
kp, des = orb.detectAndCompute(gray, None)
```

```python
import torch, torchvision
backbone = torchvision.models.resnet50(weights="DEFAULT")
backbone.fc = torch.nn.Identity()          # 2048-wymiarowy embedding
feat = backbone.eval()(x)
```

**Źródła:**
- [Distinctive Image Features from Scale-Invariant Keypoints (SIFT, Lowe)](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf)
- [OpenCV: Feature Detection and Description](https://docs.opencv.org/4.x/db/d27/tutorial_py_table_of_contents_feature2d.html)
- [Wikipedia: Histogram of oriented gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients)
- [Stanford CS231n: Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)

---

<a id="q150"></a>
### 150. Czym jest OCR i jakie są jego główne zastosowania?

**Odpowiedź:**

**OCR (Optical Character Recognition)** to automatyczne rozpoznawanie tekstu na obrazach (skany, zdjęcia, ekrany) i zamiana go na tekst cyfrowy.

#### Typowy pipeline

1. **Preprocessing**: skalowanie, binaryzacja, prostowanie (deskew), usuwanie szumu, korekta perspektywy.
2. **Detekcja tekstu (text detection)**: znalezienie regionów z tekstem (EAST, CRAFT, DBNet).
3. **Rozpoznawanie (recognition)**: zamiana wyciętego fragmentu na sekwencję znaków. Klasycznie **CRNN**: CNN + BiLSTM + **CTC loss** (dopasowanie sekwencji bez wyrównania znak-po-znaku); nowocześnie modele Transformer (TrOCR), a także end-to-end.
4. **Postprocessing**: korekta słownikowa, modele językowe, analiza układu (layout analysis), ekstrakcja pól.

Nowsze podejścia **OCR-free** (Donut) i multimodalne LLM/VLM czytają dokument bez osobnego etapu OCR.

#### Silniki

Tesseract (open source), EasyOCR, PaddleOCR, Google Cloud Vision, AWS Textract, Azure Document Intelligence.

#### Zastosowania

- Digitalizacja dokumentów i archiwów, przeszukiwalne PDF-y.
- Automatyzacja faktur, paragonów, formularzy (**Intelligent Document Processing**).
- Rozpoznawanie tablic rejestracyjnych (ANPR), dowodów, paszportów (KYC).
- Czytanie znaków drogowych, etykiet produktów, liczników.
- Dostępność (czytniki ekranu, tłumaczenie z kamery).
- Rozpoznawanie pisma odręcznego (HTR).

#### Wyzwania i metryki

Niska jakość skanów, krzywe/zakrzywione linie, pismo odręczne, fonty ozdobne, języki z diakrytykami (polskie ą, ę, ś), tabele i złożony układ. Metryki: **CER** (character error rate), **WER**, dokładność ekstrakcji pól.

```python
import pytesseract
from PIL import Image
text = pytesseract.image_to_string(Image.open("skan.png"), lang="pol")
```

**Źródła:**
- [Tesseract OCR (repozytorium)](https://github.com/tesseract-ocr/tesseract)
- [An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN)](https://arxiv.org/abs/1507.05717)
- [TrOCR: Transformer-based Optical Character Recognition](https://arxiv.org/abs/2109.10282)
- [Wikipedia: Optical character recognition](https://en.wikipedia.org/wiki/Optical_character_recognition)

---

<a id="q151"></a>
### 151. Czym CNN różni się od tradycyjnych sieci neuronowych w computer vision?

**Odpowiedź:**

**Tradycyjna sieć w pełni połączona (MLP)** wymaga spłaszczenia obrazu do wektora: obraz 224×224×3 to 150 528 wejść, a pierwsza warstwa z 1000 neuronów ma ~150 mln parametrów. Traci to strukturę przestrzenną, sieć jest podatna na overfitting i nie ma wbudowanej odporności na przesunięcia.

**CNN (Convolutional Neural Network)** wykorzystuje strukturę obrazu poprzez trzy założenia:

- **Lokalna łączność (local receptive fields)**: neuron patrzy na małe okno (np. 3×3).
- **Współdzielenie wag (weight sharing)**: ten sam filtr przesuwa się po całym obrazie, więc liczba parametrów nie zależy od rozmiaru obrazu.
- **Ekwiwariancja translacji**: przesunięcie obiektu daje przesuniętą mapę cech; pooling / stride dodaje częściową niezmienniczość.

#### Elementy

- **Konwolucja**: $y_{i,j}=\sum_{u,v}w_{u,v}\,x_{i+u,j+v}+b$
- Nieliniowość (ReLU), **pooling** (max/avg) lub stride, **normalizacja** (BatchNorm), warstwy klasyfikacyjne / global average pooling.
- Hierarchia cech: krawędzie → tekstury → części → obiekty; rosnące pole recepcyjne.
- Rozmiar wyjścia: $\lfloor (W-K+2P)/S\rfloor+1$.

#### Porównanie

| | MLP | CNN |
|---|---|---|
| Parametry | ogromna liczba | mała (współdzielone filtry) |
| Struktura 2D | ignorowana | wykorzystana |
| Inductive bias | brak | lokalność + translacja |
| Dobre dla | dane tabelaryczne | obrazy, audio, sekwencje |

Konwolucja 3×3 z 64 filtrami na 3 kanałach ma tylko $3\cdot3\cdot3\cdot64+64=1792$ parametry.

Nowoczesne architektury: LeNet, AlexNet, VGG, ResNet (połączenia rezydualne), EfficientNet, ConvNeXt. Alternatywą są **Vision Transformers**, które mają słabszy inductive bias, więc potrzebują więcej danych.

```python
import torch.nn as nn
cnn = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
    nn.Flatten(), nn.Linear(64, 10))
```

**Źródła:**
- [Stanford CS231n: Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
- [Deep Learning Book: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)

---

<a id="q152"></a>
### 152. Czym jest augmentacja danych (data augmentation) i jakie techniki są powszechnie stosowane?

**Odpowiedź:**

**Data augmentation** to sztuczne powiększanie zbioru treningowego przez losowe, zachowujące etykietę transformacje próbek. Działa jak regularyzacja: zmniejsza overfitting, uczy niezmienniczości względem nieistotnych zmian i poprawia generalizację, zwłaszcza przy małych zbiorach.

#### Techniki geometryczne

Odbicie (flip poziome; pionowe tylko gdy ma sens), obrót, przesunięcie, skalowanie, **random crop / RandomResizedCrop**, przycięcie perspektywy, transformacje afiniczne, elastic deformation (medycyna).

#### Techniki fotometryczne

Zmiana jasności, kontrastu, nasycenia, barwy (**ColorJitter**), rozmycie gaussowskie, szum, grayscale, zmiana kompresji JPEG.

#### Techniki zaawansowane

- **Cutout / Random Erasing**: zasłonięcie losowego prostokąta.
- **Mixup**: $\tilde x=\lambda x_i+(1-\lambda)x_j,\ \tilde y=\lambda y_i+(1-\lambda)y_j$.
- **CutMix**: wklejenie fragmentu jednego obrazu w drugi, etykiety mieszane proporcjonalnie do pola.
- **Mosaic** (YOLO): sklejenie 4 obrazów.
- **AutoAugment / RandAugment / TrivialAugment**: automatycznie dobierane polityki.
- **Generatywne**: GAN, modele dyfuzyjne, dane syntetyczne, copy-paste obiektów.

#### Dobre praktyki

- Augmentuj **tylko zbiór treningowy** (walidacja/test bez losowych transformacji, co najwyżej resize/normalizacja); ewentualnie TTA (test-time augmentation).
- Transformacje muszą zachować etykietę (odbicie cyfry "6" lub litery zmienia znaczenie) i być realistyczne dla domeny.
- Dla detekcji/segmentacji transformuj razem obraz i boxy/maski (Albumentations, torchvision v2).
- Augmentacja działa online (w DataLoaderze) - każda epoka widzi inne warianty.
- Zbyt silna augmentacja pogarsza wyniki (underfitting).

```python
from torchvision import transforms as T
train_tf = T.Compose([
    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3, 0.05),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

**Źródła:**
- [torchvision: Transforming and augmenting images](https://pytorch.org/vision/stable/transforms.html)
- [Albumentations documentation](https://albumentations.ai/docs/)
- [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
- [CutMix: Regularization Strategy to Train Strong Classifiers](https://arxiv.org/abs/1905.04899)
- [AutoAugment: Learning Augmentation Policies from Data](https://arxiv.org/abs/1805.09501)

---

<a id="q153"></a>
### 153. Jakie są popularne frameworki deep learning dla computer vision?

**Odpowiedź:**

#### Frameworki ogólne

- **PyTorch**: najpopularniejszy w badaniach i coraz częściej w produkcji; dynamiczny graf, `torch.compile`, bogaty ekosystem. Biblioteka **torchvision** zawiera modele (ResNet, ViT, Faster R-CNN, Mask R-CNN, DeepLab), zbiory danych i transformacje. Nakładki: PyTorch Lightning, fastai.
- **TensorFlow / Keras**: dojrzałe narzędzia produkcyjne (TF Serving, TFLite dla mobile/edge, TF.js dla przeglądarki); Keras 3 działa także na JAX i PyTorch (multi-backend).
- **JAX** (z Flax): funkcyjny, XLA, silny w badaniach na dużą skalę i TPU.

#### Biblioteki wyspecjalizowane dla CV

- **OpenCV**: klasyczne przetwarzanie obrazu, geometria, kalibracja kamer, moduł DNN do inferencji.
- **timm** (PyTorch Image Models): setki wytrenowanych backbone'ów.
- **Hugging Face Transformers / Diffusers**: ViT, DETR, SAM, CLIP, modele dyfuzyjne.
- **Detectron2** (Meta): detekcja, segmentacja, pozy.
- **Ultralytics YOLO**, **MMDetection/MMSegmentation** (OpenMMLab), **Kornia** (różniczkowalne CV), **Albumentations** (augmentacje).
- **Roboflow, CVAT, Label Studio**: dane i adnotacje.

#### Wdrożenie i optymalizacja

**ONNX / ONNX Runtime** (format wymiany), **TensorRT** (NVIDIA), **OpenVINO** (Intel), **Core ML** (Apple), **TFLite**, **NVIDIA DeepStream** (strumienie wideo).

#### Jak wybierać

- Prototypowanie i badania: PyTorch + timm/HF.
- Mobile/edge: eksport do TFLite/ONNX/Core ML z kwantyzacją.
- Klasyczna wizja i pre/postprocessing: OpenCV.
- Szybki start w detekcji: Ultralytics lub Detectron2 (sprawdź licencję).
- Ważne kryteria: społeczność, dostępność wytrenowanych modeli, wsparcie sprzętu, łatwość debugowania i wdrożenia.

```python
import timm
model = timm.create_model("convnext_tiny", pretrained=True, num_classes=10)
```

**Źródła:**
- [PyTorch torchvision documentation](https://pytorch.org/vision/stable/index.html)
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [OpenCV](https://opencv.org/)
- [Detectron2 (GitHub)](https://github.com/facebookresearch/detectron2)

---

<a id="q154"></a>
### 154. Jak Transformery mogą być używane w zadaniach computer vision?

**Odpowiedź:**

Transformery, pierwotnie z NLP, stały się podstawą nowoczesnego CV. Kluczowa trudność: obraz ma zbyt wiele pikseli na self-attention o koszcie $O(N^2)$ (224×224 = 50 176 pikseli), więc obraz dzieli się na fragmenty (**patches**).

#### Vision Transformer (ViT)

1. Obraz dzielimy na patche np. 16×16 → ok. 196 tokenów dla 224×224.
2. Każdy patch spłaszczamy i rzutujemy liniowo na embedding.
3. Dodajemy **positional embeddings** i token `[CLS]`.
4. Standardowy enkoder Transformera (self-attention + MLP).
5. Głowica klasyfikacyjna na `[CLS]`.

ViT wytrenowany na dużych zbiorach (JFT, ImageNet-21k) dorównuje lub przewyższa CNN, ale ma słabszy inductive bias (brak lokalności/translacji), więc na małych zbiorach wypada gorzej bez silnej augmentacji.

#### Rozwinięcia

- **DeiT**: efektywny trening na ImageNet z distillation.
- **Swin Transformer**: attention w oknach przesuwanych, struktura hierarchiczna, skalowalność, dobry backbone dla detekcji i segmentacji.
- **DETR**: detekcja jako predykcja zbioru; **Mask2Former, SegFormer**: segmentacja.
- **Samonadzorowane**: DINO/DINOv2, **MAE** (maskowanie 75% patchy i rekonstrukcja).
- **Multimodalne**: **CLIP** (kontrastowe dopasowanie obraz-tekst, zero-shot), VLM, **SAM**, modele dyfuzyjne z DiT.
- **Wideo**: ViViT, TimeSformer.

#### Zalety i wady

- Zalety: globalny kontekst od pierwszej warstwy, skalowanie z danymi i mocą obliczeniową, jednolita architektura dla modalności.
- Wady: duże zapotrzebowanie na dane i obliczenia, kwadratowy koszt w liczbie tokenów, wolniejsze na edge (stąd hybrydy CNN+Transformer, MobileViT).

```python
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
```

**Źródła:**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT)](https://arxiv.org/abs/2010.11929)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)
- [Training data-efficient image transformers (DeiT)](https://arxiv.org/abs/2012.12877)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)

---

## Duże modele językowe (LLM)

<a id="q155"></a>
### 155. Czym jest duży model językowy (LLM) i jak działa?

**Odpowiedź:**

**LLM (Large Language Model)** to sieć neuronowa (prawie zawsze Transformer typu decoder-only) o miliardach parametrów, wytrenowana na ogromnych korpusach tekstu do **przewidywania następnego tokenu**. Mimo prostego celu model uczy się gramatyki, faktów, wnioskowania i kodu, a wraz ze skalą pojawiają się nowe zdolności (in-context learning, chain-of-thought).

#### Jak działa

1. **Tokenizacja**: tekst dzielony na tokeny (BPE / SentencePiece), każdy token to id.
2. **Embedding** + informacja pozycyjna (RoPE).
3. **Stos bloków Transformera**: masked self-attention (token widzi tylko poprzednie) + FFN, z residual connections i normalizacją.
4. **Głowica LM**: logity po słowniku → softmax:
$$p(x_t\mid x_{<t})=\mathrm{softmax}(W h_t)$$
5. **Generowanie autoregresyjne**: wybieramy token (greedy, temperature, top-k, top-p, beam search), doklejamy i powtarzamy. **KV cache** przyspiesza kolejne kroki.

Funkcja straty: cross-entropy $\mathcal{L}=-\sum_t\log p(x_t\mid x_{<t})$.

#### Etapy tworzenia

- **Pre-training** na bilionach tokenów (samonadzorowane).
- **Supervised fine-tuning (SFT)** na instrukcjach i odpowiedziach.
- **Alignment**: RLHF, DPO, Constitutional AI - dopasowanie do preferencji i bezpieczeństwa.

#### Rozszerzenia (Inżynieria AI)

- **RAG**: dostarczanie kontekstu z zewnętrznej bazy wiedzy, ograniczające halucynacje.
- **Agenci i MCP**: LLM wywołuje narzędzia (function calling), planuje wieloetapowe zadania.
- **Fine-tuning parametrycznie efektywny** (LoRA/QLoRA) i **kwantyzacja** (INT8/INT4) dla tańszego uruchamiania.

#### Ograniczenia

Halucynacje, ograniczone okno kontekstu, brak aktualnej wiedzy bez retrievalu, uprzedzenia, koszt obliczeń, wrażliwość na prompt, ryzyka bezpieczeństwa (prompt injection).

**Źródła:**
- [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization (YouTube, Outcome School)](https://www.youtube.com/watch?v=lnfWvX66FUk)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)

---

<a id="q156"></a>
### 156. Czym jest architektura Transformer i jak działa?

**Odpowiedź:**

**Transformer** (Vaswani i in., 2017, "Attention Is All You Need") to architektura sekwencyjna oparta wyłącznie na mechanizmie **attention**, bez rekurencji i konwolucji. Pozwala przetwarzać wszystkie pozycje równolegle i bezpośrednio modelować zależności dalekiego zasięgu.

#### Przepływ danych

1. **Tokeny → embeddingi** wymiaru $d_{model}$ + **kodowanie pozycyjne** (sinusoidalne w oryginale; RoPE/ALiBi w nowszych), bo attention samo w sobie nie zna kolejności.
2. **Stos $N$ identycznych bloków**, każdy zawiera:
   - **Multi-head self-attention**,
   - **Feed-forward network (FFN)** stosowany pozycja po pozycji,
   - **residual connections** + **LayerNorm**.
3. Wyjście: reprezentacje kontekstowe każdego tokenu.

#### Self-attention

Dla macierzy wejściowej $X$: $Q=XW_Q,\ K=XW_K,\ V=XW_V$ i
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
Każdy token liczy wagi względem wszystkich innych i bierze ich ważoną sumę wartości.

#### Warianty

- **Encoder-decoder** (oryginał, T5, tłumaczenie): enkoder z dwukierunkowym attention, dekoder z maskowanym self-attention i **cross-attention** do enkodera.
- **Encoder-only** (BERT): rozumienie tekstu, klasyfikacja.
- **Decoder-only** (GPT, Llama): generowanie; **causal mask** blokuje patrzenie w przyszłość.

#### Dlaczego zwyciężył

Pełna równoległość treningu (RNN są sekwencyjne), krótka ścieżka między dowolnymi tokenami (łatwiejszy przepływ gradientu), doskonała skalowalność. Koszt: attention $O(n^2)$ względem długości sekwencji (stąd FlashAttention, sparse/linear attention).

**Źródła:**
- [Decoding Transformer Architecture (Outcome School)](https://outcomeschool.com/blog/decoding-transformer-architecture)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/)

---

<a id="q157"></a>
### 157. Jakie są kluczowe komponenty architektury Transformer?

**Odpowiedź:**

#### 1. Tokenizacja i embeddingi wejściowe

Tokeny (subwords) mapowane na wektory $d_{model}$ przez tablicę embeddingów. W modelach decoder-only macierz wyjściowa bywa współdzielona z embeddingiem (weight tying).

#### 2. Kodowanie pozycyjne

Wstrzykuje informację o kolejności. Oryginał: sinusoidy $PE_{(pos,2i)}=\sin(pos/10000^{2i/d})$. Nowsze: **RoPE** (rotacja Q/K zależna od pozycji), **ALiBi** (bias w attention), uczone embeddingi pozycji.

#### 3. Multi-Head Attention (MHA)

$h$ głowic z osobnymi projekcjami $W_Q^i,W_K^i,W_V^i$ (wymiar $d_k=d_{model}/h$); wyniki konkatenowane i rzutowane przez $W_O$. Różne głowice uczą się różnych relacji (składnia, koreferencja, pozycja). Warianty: **MQA/GQA** (współdzielone K/V dla oszczędności KV cache).

#### 4. Maskowanie

- **Padding mask** - ignorowanie tokenów wypełniających.
- **Causal mask** - w dekoderze, pozycja $t$ nie widzi $t+1,\dots$.

#### 5. Feed-Forward Network (FFN)

Dwuwarstwowy MLP na każdej pozycji: $\mathrm{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2$, zwykle o wymiarze pośrednim ~4$d_{model}$ (dziś często **SwiGLU**). Zawiera większość parametrów; w MoE jest zastępowany zestawem ekspertów.

#### 6. Residual connections i normalizacja

$x\leftarrow x+\mathrm{Sublayer}(\mathrm{Norm}(x))$. Oryginał: post-LN; nowoczesne modele **pre-LN** i **RMSNorm** (stabilniejszy trening głębokich sieci).

#### 7. Cross-attention (tylko encoder-decoder)

Zapytania z dekodera, klucze i wartości z wyjścia enkodera.

#### 8. Głowica wyjściowa

Warstwa liniowa + softmax po słowniku (LM) lub głowica zadaniowa (klasyfikacja).

#### Regularyzacja i trening

Dropout, label smoothing, warm-up learning rate + decay, Adam(W).

**Źródła:**
- [Decoding Transformer Architecture (Outcome School)](https://outcomeschool.com/blog/decoding-transformer-architecture)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [GLU Variants Improve Transformer (SwiGLU)](https://arxiv.org/abs/2002.05202)

---

<a id="q158"></a>
### 158. Czego uczy się każdy blok Transformera (What Each Transformer Block Learns)?

**Odpowiedź:**

Nie ma ścisłej, jednoznacznej odpowiedzi, ale badania interpretowalności i probing dają spójny obraz **hierarchii abstrakcji**: kolejne bloki budują coraz bardziej złożone reprezentacje, a każdy blok składa się z dwóch pod-warstw pełniących różne role.

#### Dwie pod-warstwy w bloku

- **Attention** - *przenosi informację między tokenami*: decyduje, skąd zebrać kontekst. Głowice wyspecjalizowane w np. następnym/poprzednim tokenie, koreferencji, relacjach składniowych (podmiot-orzeczenie), kopiowaniu (**induction heads** umożliwiające in-context learning).
- **FFN (MLP)** - *przetwarza informację w obrębie tokenu*: działa jak pamięć klucz-wartość, przechowując wiedzę faktograficzną i wzorce; neurony aktywują się na konkretne cechy/koncepty (Geva i in.).
- Strumień residual pełni rolę wspólnej "szyny": każdy blok dodaje do niego swoją poprawkę.

#### Hierarchia po głębokości (dla enkoderów typu BERT)

| Warstwy | Typowe informacje |
|---|---|
| Dolne | morfologia, lokalna składnia, część mowy, powierzchowne cechy |
| Środkowe | struktura składniowa, relacje składniowe, drzewa zależności |
| Górne | semantyka, koreferencja, cechy zależne od zadania |

Badania "BERT Rediscovers the Classical NLP Pipeline" pokazują, że kolejność ta odpowiada klasycznemu pipeline: POS → parsing → NER → role semantyczne → koreferencja. W dużych LLM wczesne warstwy budują reprezentacje leksykalne, środkowe wiedzę i wnioskowanie, a ostatnie przygotowują rozkład następnego tokenu. Wykorzystuje to np. fine-tuning tylko górnych warstw lub **logit lens**.

#### Zastrzeżenia

- Podział jest przybliżony i zależy od modelu i zadania; funkcje są rozproszone (superpozycja cech).
- Wyniki z probingu pokazują, co jest *dekodowalne*, nie zawsze co model *używa*.
- Ostatnie bloki często są mniej istotne (możliwy pruning warstw), stąd badania nad early exit.

Praktycznie: w transfer learningu dolne warstwy bywają zamrażane, a wyższe strojone pod zadanie.

**Źródła:**
- [Decoding Transformer Architecture (Outcome School)](https://outcomeschool.com/blog/decoding-transformer-architecture)
- [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950)
- [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913)
- [A Mathematical Framework for Transformer Circuits (Anthropic)](https://transformer-circuits.pub/2021/framework/index.html)

---

<a id="q159"></a>
### 159. Dlaczego skalujemy iloczyn skalarny w attention przez √dₖ?

**Odpowiedź:**

W scaled dot-product attention:
$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
Skalowanie chroni softmax przed **nasyceniem** przy dużym wymiarze $d_k$.

#### Uzasadnienie statystyczne

Załóżmy, że składniki wektorów $q,k\in\mathbb{R}^{d_k}$ są niezależne, o średniej 0 i wariancji 1. Wtedy iloczyn $q\cdot k=\sum_{i=1}^{d_k}q_ik_i$ ma:

- $\mathbb{E}[q\cdot k]=0$,
- $\mathrm{Var}(q\cdot k)=\sum_i\mathrm{Var}(q_ik_i)=d_k$ (bo $\mathrm{Var}(q_ik_i)=\mathbb{E}[q_i^2]\mathbb{E}[k_i^2]=1$).

Odchylenie standardowe rośnie więc jak $\sqrt{d_k}$. Dzieląc przez $\sqrt{d_k}$ przywracamy wariancję 1, niezależnie od wymiaru.

#### Co się dzieje bez skalowania

Dla dużego $d_k$ (np. 64-128) logity mają duży rozrzut, więc softmax staje się bardzo "ostry" (prawie one-hot). Wtedy:

- gradient softmax $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$ dąży do zera (małe $p_i(1-p_i)$),
- uczenie jest wolne lub niestabilne (znikające gradienty),
- model nie potrafi "rozmywać" uwagi na wiele tokenów na początku treningu.

#### Przykład

Dla $d_k=64$ nieskalowany iloczyn ma odchylenie ~8, więc różnice logitów rzędu kilkunastu dają wagi ~0 lub ~1. Po podzieleniu przez 8 odchylenie wynosi ~1 i softmax pozostaje w reżimie z użytecznymi gradientami. Analogicznie działa temperatura w softmax: $\sqrt{d_k}$ to stała temperatura.

#### Uwagi

- Wybór $\sqrt{d_k}$ zakłada niezależność i jednostkową wariancję; przy inicjalizacji i normalizacjach to przybliżenie.
- Niektóre modele stosują dodatkowo QK-norm dla stabilności przy dużych skalach.

**Źródła:**
- [Math behind √dₖ Scaling Factor in Attention (Outcome School)](https://outcomeschool.com/blog/scaling-dot-product-attention)
- [Attention Is All You Need (sekcja 3.2.1)](https://arxiv.org/abs/1706.03762)
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

---

<a id="q160"></a>
### 160. Czym jest KV Cache w LLM?

**Odpowiedź:**

**KV cache** to mechanizm przyspieszający generowanie autoregresyjne: zamiast przeliczać klucze (K) i wartości (V) wszystkich poprzednich tokenów w każdym kroku, zapamiętujemy je i dla nowego tokenu liczymy tylko jego Q, K, V.

#### Problem bez cache

Generując token $t$, attention potrzebuje K i V wszystkich tokenów $1..t$. W naiwnej implementacji przeliczamy je od nowa, co daje koszt rosnący kwadratowo w liczbie kroków. Ponieważ maska kauzalna sprawia, że K/V wcześniejszych tokenów **nigdy się nie zmieniają**, można je bezpiecznie przechować.

#### Dwie fazy inferencji

- **Prefill**: przetworzenie całego promptu równolegle, wypełnienie cache (compute-bound).
- **Decode**: token po tokenie, każdy krok dodaje jedną parę K/V do cache (memory-bandwidth-bound).

#### Zużycie pamięci

$$\text{KV bytes}=2\cdot n_{layers}\cdot n_{kv\_heads}\cdot d_{head}\cdot L\cdot B\cdot \text{bytes}_{per\_elem}$$
Czynnik 2 = K i V, $L$ = długość kontekstu, $B$ = rozmiar batcha. Rośnie liniowo z kontekstem i batchem i przy długich kontekstach może przewyższyć rozmiar wag modelu, stąd ograniczenie przepustowości i liczby równoległych użytkowników.

#### Techniki redukcji

- **MQA / GQA**: mniej głowic K/V niż Q (dziś standard, np. w modelach Llama).
- **PagedAttention** (vLLM): stronicowanie cache eliminujące fragmentację.
- **Kwantyzacja KV cache** (INT8/FP8), **sliding window attention**, eviction/kompresja tokenów, **prefix caching** (współdzielenie cache wspólnych promptów), **MLA** (DeepSeek).

```python
out = model.generate(**inputs, max_new_tokens=100, use_cache=True)  # domyślnie True
```

**Źródła:**
- [KV Cache in LLMs (Outcome School)](https://outcomeschool.com/blog/kv-cache-in-llms)
- [Hugging Face: KV cache strategies](https://huggingface.co/docs/transformers/kv_cache)
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245)
- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150)

---

<a id="q161"></a>
### 161. Czym jest Paged Attention w LLM?

**Odpowiedź:**

**PagedAttention** to algorytm zarządzania pamięcią KV cache zaproponowany w systemie **vLLM**, inspirowany stronicowaniem pamięci wirtualnej w systemach operacyjnych.

#### Problem

KV cache każdego żądania rośnie dynamicznie i ma nieznaną z góry długość. Klasyczne serwery rezerwowały ciągły blok pamięci na maksymalną długość, co powodowało:

- **fragmentację wewnętrzną** (niewykorzystana rezerwacja),
- **fragmentację zewnętrzną**,
- brak współdzielenia cache między żądaniami/sekwencjami.

W efekcie duża część pamięci GPU szła na straty (autorzy raportują, że w istniejących systemach marnowane było około 60-80% pamięci KV cache), co ograniczało rozmiar batcha i przepustowość.

#### Rozwiązanie

- KV cache dzielimy na **bloki (strony)** o stałym rozmiarze (np. 16 tokenów).
- Bloki logiczne sekwencji mapowane są na **niekoniecznie ciągłe** bloki fizyczne przez **block table** (jak tablica stron).
- Bloki alokowane **na żądanie**; marnowany jest co najwyżej niepełny ostatni blok.
- Kernel attention czyta K/V przez tablicę bloków.

#### Zalety

- Prawie brak fragmentacji, więc większe batche i wyższa przepustowość (autorzy vLLM raportują kilkukrotną poprawę względem wcześniejszych systemów).
- **Współdzielenie z copy-on-write**: wspólny prefiks (system prompt, parallel sampling, beam search) przechowywany raz; kopiowanie bloku dopiero przy rozejściu sekwencji.
- Naturalne wsparcie dla prefix caching, preemption (swap do CPU).

#### Koszt

Nieco bardziej złożony kernel (indirekcja przez block table), ale narzut jest mały w porównaniu z zyskiem. Łączy się z **continuous batching**.

**Źródła:**
- [Paged Attention in LLMs (Outcome School)](https://outcomeschool.com/blog/paged-attention-in-llms)
- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM documentation](https://docs.vllm.ai/)

---

<a id="q162"></a>
### 162. Czym jest Flash Attention?

**Odpowiedź:**

**FlashAttention** (Dao i in., 2022) to **dokładna** (nie przybliżona) implementacja attention, która jest szybsza i zużywa mniej pamięci dzięki uwzględnieniu hierarchii pamięci GPU (**IO-aware**).

#### Problem: wąskim gardłem jest pamięć, nie FLOP-y

Standardowe attention materializuje macierz $S=QK^\top\in\mathbb{R}^{N\times N}$ i $P=\mathrm{softmax}(S)$ w wolnej pamięci **HBM**, wielokrotnie ją czytając i zapisując. Pamięć rośnie kwadratowo ($O(N^2)$), a czas zdominowany jest przez transfery HBM ↔ SRAM.

#### Idea

1. **Tiling**: dzielimy Q, K, V na bloki mieszczące się w szybkiej pamięci **SRAM** (on-chip) i liczymy attention blok po bloku w jednym skernelowanym przebiegu (**kernel fusion**).
2. **Online softmax**: softmax obliczany przyrostowo, ze śledzeniem maksimum $m$ i sumy normalizującej $\ell$ na wiersz, więc nie potrzeba całej macierzy $S$:
$$m^{new}=\max(m,\tilde m),\quad \ell^{new}=e^{m-m^{new}}\ell+e^{\tilde m-m^{new}}\tilde\ell$$
3. **Recomputation w backward**: zamiast zapisywać $N\times N$ macierz, przelicza się ją z bloków (kompromis: więcej FLOP-ów, mniej I/O, ale szybciej w sumie).

#### Efekty

- Pamięć **liniowa** w długości sekwencji $O(N)$ zamiast $O(N^2)$.
- Wynik numerycznie równoważny standardowemu attention.
- Zwykle kilkukrotne przyspieszenie treningu i inferencji oraz możliwość używania dłuższych kontekstów. Konkretny zysk zależy od GPU i długości sekwencji.
- **FlashAttention-2** poprawia podział pracy i równoległość; **FlashAttention-3** wykorzystuje cechy GPU Hopper (asynchroniczność, FP8).

#### Uwagi praktyczne

Czas obliczeń nadal $O(N^2)$. W PyTorch dostępny przez `torch.nn.functional.scaled_dot_product_attention` (automatyczny wybór backendu), w HF przez `attn_implementation="flash_attention_2"`.

```python
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**Źródła:**
- [Decoding Flash Attention in LLMs (Outcome School)](https://outcomeschool.com/blog/decoding-flash-attention)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)

---

<a id="q163"></a>
### 163. Czym jest speculative decoding i jak przyspiesza inferencję?

**Odpowiedź:**

**Speculative decoding** to technika przyspieszania generowania LLM **bez zmiany rozkładu wyjścia** (wynik jest statystycznie identyczny jak z modelu docelowego).

#### Problem

Dekodowanie autoregresyjne jest ograniczone przepustowością pamięci: każdy token wymaga załadowania wszystkich wag, a GPU jest niedociążony. Weryfikacja wielu tokenów naraz kosztuje niewiele więcej niż wygenerowanie jednego.

#### Algorytm

1. Mały, szybki **model roboczy (draft)** $q$ generuje $\gamma$ kolejnych tokenów-kandydatów.
2. Duży **model docelowy (target)** $p$ przelicza wszystkie kandydaty **w jednym przebiegu równoległym** i zwraca $p(x_i\mid\cdot)$ dla każdej pozycji.
3. Dla kolejnych tokenów stosujemy **rejection sampling**: token $x$ jest akceptowany z prawdopodobieństwem
$$\min\!\left(1,\frac{p(x)}{q(x)}\right)$$
4. Przy pierwszym odrzuceniu próbkujemy token z rozkładu poprawionego $\propto\max(0,p-q)$ i porzucamy resztę kandydatów; jeśli wszystkie zaakceptowano, dostajemy dodatkowy token z $p$.

Dzięki temu w jednym kroku dużego modelu powstaje średnio kilka tokenów, a rozkład końcowy pozostaje dokładnie rozkładem $p$.

#### Od czego zależy przyspieszenie

- **Acceptance rate** (podobieństwo draftu do targetu): wyższy = lepiej.
- Stosunek kosztu draftu do targetu i liczba kandydatów $\gamma$.
- Zwykle zysk 2-3x, największy przy przewidywalnym tekście (kod, format), mniejszy przy dużej losowości; przy dużych batchach (compute-bound) zysk maleje.

#### Warianty

- Draft jako mniejszy model z tej samej rodziny, **self-speculation** (early exit, **Medusa** - dodatkowe głowice, **EAGLE**), **prompt lookup / n-gram**, drzewa kandydatów (tree attention).
- Koszt: dodatkowa pamięć na draft i złożoność serwowania; dostępne w vLLM, TensorRT-LLM, HF (`assistant_model`).

**Źródła:**
- [Speculative Decoding (Outcome School)](https://outcomeschool.com/blog/speculative-decoding)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318)

---

<a id="q164"></a>
### 164. Jak continuous batching poprawia przepustowość inferencji LLM?

**Odpowiedź:**

#### Problem z batchingiem statycznym

W klasycznym (statycznym) batchingu grupujemy żądania i przetwarzamy je razem aż **wszystkie** skończą generowanie. Ponieważ długości odpowiedzi różnią się, krótkie sekwencje kończą się wcześniej, a ich miejsca w batchu stoją puste (lub generują padding), dopóki nie skończy najdłuższa. Efekt: niski utylizowany procent GPU, wysoka latencja dla nowych żądań czekających na następny batch.

#### Continuous (in-flight, iteration-level) batching

Harmonogram działa na poziomie **pojedynczej iteracji dekodowania** (jednego tokenu), a nie całego żądania:

1. W każdej iteracji przetwarzamy aktualny zestaw aktywnych sekwencji.
2. Gdy sekwencja się kończy (EOS lub limit), jest natychmiast usuwana z batcha.
3. Na jej miejsce **od razu wchodzi** oczekujące żądanie (jego prefill, ewentualnie porcjowany - *chunked prefill*).

Batch cały czas jest "pełny", a nowe żądania nie czekają na koniec poprzednich.

#### Korzyści

- Znacznie wyższa **przepustowość** (tokeny/s) i lepsze wykorzystanie GPU; Orca pokazała wielokrotne przyspieszenie względem statycznego batchingu, a późniejsze systemy potwierdzają duże zyski.
- Niższa **latencja** kolejkowania (time-to-first-token).
- Naturalne dopasowanie do nieprzewidywalnych długości odpowiedzi.

#### Wymagania i pułapki

- Wymaga elastycznego zarządzania pamięcią KV cache, stąd połączenie z **PagedAttention**.
- Dochodzi problem interferencji prefill vs decode (długi prefill spowalnia dekodowanie innych) - rozwiązania: chunked prefill, rozdzielenie prefill/decode (disaggregation).
- Kompromis latencja-przepustowość sterujemy maksymalnym rozmiarem batcha i polityką kolejkowania (limit tokenów w iteracji).

Continuous batching jest standardem w vLLM, TGI, TensorRT-LLM, SGLang.

**Źródła:**
- [Continuous Batching in LLMs (Outcome School)](https://outcomeschool.com/blog/continuous-batching-in-llms)
- [Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI '22)](https://www.usenix.org/conference/osdi22/presentation/yu)
- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)

---

<a id="q165"></a>
### 165. Czym różni się pre-training od fine-tuningu w LLM?

**Odpowiedź:**

#### Pre-training

- **Cel**: nauczenie modelu ogólnej wiedzy o języku i świecie.
- **Dane**: ogromne, nieoznaczone korpusy (web, książki, kod) - bilony tokenów.
- **Zadanie**: samonadzorowane, zwykle next-token prediction (GPT/Llama) lub masked language modeling (BERT).
- **Koszt**: ekstremalny (tysiące GPU, tygodnie-miesiące), robi to niewiele organizacji.
- **Wynik**: model bazowy (*base model*), dobry w kontynuowaniu tekstu, ale nie w wykonywaniu poleceń.

#### Fine-tuning

Dalsze uczenie modelu bazowego na mniejszym, dopasowanym zbiorze:

- **SFT (Supervised / Instruction tuning)**: pary instrukcja-odpowiedź, zamiana modelu bazowego w asystenta.
- **Alignment**: RLHF, DPO - dopasowanie do preferencji ludzi (pomocność, bezpieczeństwo).
- **Domain adaptation / continued pre-training**: dalszy pre-training na danych domenowych (medycyna, prawo).
- **Task-specific fine-tuning**: klasyfikacja, ekstrakcja, styl.

#### Metody fine-tuningu

| Metoda | Opis |
|---|---|
| Full fine-tuning | aktualizacja wszystkich wag; drogie, ryzyko catastrophic forgetting |
| **PEFT: LoRA / QLoRA** | uczymy małe macierze niskiego rzędu (adaptery) $W+BA$, 4-bit model bazowy w QLoRA; tanie, mały plik adaptera |
| Prompt/prefix tuning | uczymy wirtualne tokeny |

#### Porównanie

| | Pre-training | Fine-tuning |
|---|---|---|
| Dane | ogromne, nieoznaczone | małe (tysiące-miliony), często oznaczone |
| Koszt | bardzo wysoki | niski-umiarkowany |
| Cel | wiedza ogólna | zachowanie/zadanie/domena |

#### Kiedy co

Zwykle najpierw próbuj **prompt engineeringu i RAG** (aktualna wiedza, cytowania); fine-tuning wybierz dla stylu, formatu, specjalistycznych zadań, obniżenia kosztu/latencji (mniejszy model) lub gdy prompt nie wystarcza. Pułapki: overfitting na małym zbiorze, forgetting, jakość danych ważniejsza niż ilość.

**Źródła:**
- [AI Engineering Explained: LLM, RAG, MCP, Agent, Fine-Tuning, Quantization (YouTube)](https://www.youtube.com/watch?v=lnfWvX66FUk)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Hugging Face: Fine-tune a pretrained model](https://huggingface.co/docs/transformers/training)

---

<a id="q166"></a>
### 166. Jakie są wyzwania w trenowaniu LLM?

**Odpowiedź:**

#### Obliczenia i koszt

Trening dużego modelu wymaga tysięcy GPU/TPU przez tygodnie i budżetów rzędu milionów dolarów. Skalowanie regułami **scaling laws** (Kaplan; **Chinchilla**: liczba tokenów powinna rosnąć proporcjonalnie do liczby parametrów, ok. 20 tokenów na parametr jako reguła kciuka dla optymalności obliczeniowej). Coraz częściej trenuje się mniejsze modele na większej liczbie tokenów, by obniżyć koszt inferencji.

#### Pamięć i równoległość

Wagi + gradienty + stan optymalizatora (Adam ~ 2 dodatkowe kopie) + aktywacje nie mieszczą się na jednym GPU. Rozwiązania:

- **Data parallelism**, **tensor parallelism**, **pipeline parallelism**, sequence/context parallelism,
- **ZeRO / FSDP** (shardowanie stanu optymalizatora, gradientów i parametrów),
- **Mixed precision** (BF16/FP16/FP8), gradient checkpointing, offloading,
- efektywna komunikacja (NVLink, InfiniBand); narzut synchronizacji obniża wykorzystanie sprzętu (MFU).

#### Stabilność treningu

Loss spikes, dywergencja, niestabilność numeryczna (FP16 overflow), wrażliwość na learning rate i inicjalizację. Środki: warm-up, cosine decay, gradient clipping, pre-LN/RMSNorm, BF16, restart z checkpointu, monitorowanie norm gradientów. Awarie sprzętu przy dużych klastrach wymagają częstych checkpointów i automatycznego wznawiania.

#### Dane

- **Jakość > ilość**: filtrowanie, deduplikacja (exact i near-duplicate), usuwanie toksyczności i PII.
- **Kontaminacja benchmarków** (dane testowe w zbiorze treningowym).
- Ograniczona ilość wysokiej jakości tekstu w internecie, dane syntetyczne i ryzyko *model collapse*.
- Miks danych (kod, matematyka, języki), kwestie licencji i praw autorskich.

#### Ewaluacja i bezpieczeństwo

Trudność pomiaru zdolności, benchmarki nasycają się, halucynacje, uprzedzenia, alignment (RLHF/DPO), ryzyko wycieku danych treningowych (memorization).

#### Inne

Wpływ środowiskowy (energia), wielojęzyczność (języki niskozasobowe, np. polski ma mniej danych), długi kontekst (koszt attention $O(n^2)$), odtwarzalność.

**Źródła:**
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)

---

<a id="q167"></a>
### 167. Czym jest zero-shot learning w kontekście LLM?

**Odpowiedź:**

**Zero-shot learning** w LLM oznacza wykonanie zadania **bez żadnych przykładów** w prompcie i bez dodatkowego trenowania: model dostaje tylko opis zadania (instrukcję) i dane wejściowe, a polega na wiedzy zdobytej w pre-trainingu i instruction tuningu.

#### Porównanie ze skalą przykładów

| Podejście | Co dostaje model |
|---|---|
| **Zero-shot** | sama instrukcja |
| **One-shot / few-shot** | instrukcja + 1 / kilka przykładów w prompcie (in-context learning) |
| **Fine-tuning** | aktualizacja wag na przykładach |

#### Przykład

```
Sklasyfikuj sentyment recenzji jako pozytywny, negatywny lub neutralny.
Recenzja: "Dostawa spóźniona, ale produkt świetny."
Sentyment:
```

Model odpowiada bez wcześniejszych przykładów.

#### Dlaczego to działa

- Pre-training na różnorodnym tekście sprawia, że zadania pojawiają się implicite w danych.
- **Instruction tuning** (FLAN, T0, InstructGPT) wyraźnie poprawia zero-shot: model uczy się rozumieć polecenia także dla nowych zadań.
- Zdolność rośnie ze skalą (GPT-3 pokazał, że zero-/few-shot poprawia się z rozmiarem).

#### Warianty i techniki

- **Zero-shot Chain-of-Thought**: dopisek "Let's think step by step" poprawia wnioskowanie.
- Zero-shot w embeddingach/CLIP: klasyfikacja przez porównanie z opisami klas.
- Zero-shot klasyfikacja z NLI (`zero-shot-classification` w HF).

#### Ograniczenia

Wrażliwość na sformułowanie promptu, niestabilny format wyjścia, gorsza jakość niż few-shot na zadaniach niszowych, halucynacje. Poprawa: dodać przykłady (few-shot), sprecyzować format (np. JSON schema), użyć RAG lub fine-tuningu.

**Źródła:**
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652)
- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)

---

<a id="q168"></a>
### 168. Jak radzić sobie z uprzedzeniami (bias) i sprawiedliwością (fairness) w LLM?

**Odpowiedź:**

#### Skąd biorą się uprzedzenia

- **Dane treningowe**: internet odzwierciedla stereotypy, nierównomierną reprezentację grup, języków i kultur.
- **Etykiety i preferencje** (annotatorzy, RLHF) niosą własne skrzywienia.
- **Modelowanie i dekodowanie**: model może wzmacniać korelacje statystyczne.
- **Zastosowanie**: kontekst użycia (rekrutacja, kredyty) zmienia wagę szkód (szkody alokacyjne i reprezentacyjne).

#### Pomiar

- Benchmarki: **BBQ** (Bias Benchmark for QA), **StereoSet**, **CrowS-Pairs**, **RealToxicityPrompts**, **WinoBias**.
- Testy kontrfaktyczne: zamiana atrybutu chronionego (imię, płeć, pochodzenie) w tym samym prompcie i porównanie odpowiedzi.
- Metryki grupowe: parytet demograficzny, equalized odds, różnica FPR/TPR między grupami; red teaming; audyty ludzi.

#### Mitygacja (na każdym etapie)

1. **Dane**: kuracja i filtrowanie, zbalansowanie reprezentacji, augmentacja kontrfaktyczna, dokumentacja (datasheets, model cards).
2. **Trening**: alignment (RLHF/DPO, Constitutional AI), fine-tuning na zbalansowanych danych, celowe rozszerzenie danych dla języków niskozasobowych.
3. **Inferencja**: system prompt z zasadami, filtry wejścia/wyjścia (guardrails, klasyfikatory toksyczności), kalibracja odpowiedzi.
4. **Proces**: różnorodne zespoły annotatorów, ciągły monitoring w produkcji, mechanizmy zgłaszania błędów, human-in-the-loop dla decyzji wysokiego ryzyka.

#### Kompromisy i trudności

- Różne definicje sprawiedliwości bywają matematycznie sprzeczne (nie można spełnić ich wszystkich naraz).
- Redukcja bias może obniżyć użyteczność lub prowadzić do nadmiernej odmowy.
- Bias ukryty (implicit) trudniej wykryć; wynik zależy od języka (uprzedzenia w polskim inne niż w angielskim) i kultury.
- Uwaga na zgodność regulacyjną (EU AI Act, RODO) i przejrzystość.

**Źródła:**
- [Ethical and social risks of harm from Language Models (Weidinger i in.)](https://arxiv.org/abs/2112.04359)
- [BBQ: A Hand-Built Bias Benchmark for Question Answering](https://arxiv.org/abs/2110.08193)
- [RealToxicityPrompts](https://arxiv.org/abs/2009.11462)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://arxiv.org/abs/1607.06520)

---

<a id="q169"></a>
### 169. Jakie są rzeczywiste zastosowania LLM w biznesie i technologii?

**Odpowiedź:**

#### Obsługa klienta i sprzedaż

Chatboty i asystenci (rozwiązywanie zgłoszeń, FAQ), podsumowania rozmów, sugestie odpowiedzi dla konsultantów, personalizacja ofert i e-maili, kwalifikacja leadów.

#### Wiedza i wyszukiwanie

**RAG** nad dokumentacją firmową, wewnętrzne wyszukiwarki semantyczne, Q&A nad umowami i politykami, streszczanie raportów i spotkań.

#### Rozwój oprogramowania

Uzupełnianie i generowanie kodu (Copilot, Codex), refaktoryzacja, generowanie testów i dokumentacji, wyjaśnianie legacy code, review, agenci kodujący wykonujący wieloetapowe zadania w repozytorium.

#### Przetwarzanie dokumentów i danych

Ekstrakcja informacji (faktury, umowy, CV) do JSON, klasyfikacja i routing, tłumaczenia, moderacja treści, text-to-SQL i analityka w języku naturalnym, czyszczenie i wzbogacanie danych.

#### Treści i marketing

Copywriting, lokalizacja, szkice raportów, generowanie opisów produktów, materiałów szkoleniowych.

#### Branże regulowane

- **Finanse**: analiza raportów, compliance, wykrywanie nadużyć (wspierająco).
- **Medycyna**: dokumentacja kliniczna, streszczanie kart, wsparcie decyzji (pod nadzorem lekarza).
- **Prawo**: przegląd umów, wyszukiwanie orzecznictwa.
- **Edukacja**: korepetytor, generowanie ćwiczeń.

#### Agenci i automatyzacja

Agenci wywołujący narzędzia (function calling, MCP): rezerwacje, workflow, automatyzacja procesów back-office.

#### Wdrożenie: na co uważać

- Halucynacje - stosuj RAG, cytowania, walidację wyjścia, human-in-the-loop.
- Prywatność i bezpieczeństwo danych (RODO, wyciek przez prompt, prompt injection).
- Koszt i latencja (mniejsze modele, cache, distillation, kwantyzacja).
- Ewaluacja na własnych danych (golden set, LLM-as-judge z kalibracją), monitoring jakości.
- Mierzenie ROI: czas obsługi, deflekcja zgłoszeń, produktywność deweloperów.

**Źródła:**
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Evaluating Large Language Models Trained on Code (Codex)](https://arxiv.org/abs/2107.03374)
- [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564)

---

<a id="q170"></a>
### 170. W jaki sposób architektura Transformer poprawia wydajność LLM względem RNN?

**Odpowiedź:**

#### Ograniczenia RNN/LSTM

- **Przetwarzanie sekwencyjne**: stan $h_t=f(h_{t-1},x_t)$ zależy od poprzedniego, więc trening nie da się zrównoleglić po osi czasu, słabo wykorzystuje GPU.
- **Zależności dalekiego zasięgu**: informacja musi przejść przez wiele kroków; gradient przechodzi przez iloczyn jakobianów, co powoduje **zanikanie/eksplozję gradientów** (LSTM/GRU łagodzą, ale nie eliminują).
- **Wąskie gardło stanu**: cała historia skompresowana do stanu stałej wielkości.
- Trudność skalowania do miliardów parametrów i bilionów tokenów.

#### Jak Transformer to rozwiązuje

| Aspekt | RNN | Transformer |
|---|---|---|
| Równoległość treningu | brak (sekwencyjnie po krokach) | pełna po całej sekwencji |
| Długość ścieżki między tokenami | $O(n)$ | $O(1)$ (bezpośredni attention) |
| Pamięć historii | skompresowany stan | wszystkie tokeny (KV) |
| Złożoność warstwy | $O(n\cdot d^2)$ | $O(n^2\cdot d)$ |
| Skalowalność | ograniczona | bardzo dobra (scaling laws) |

- **Self-attention** daje bezpośredni dostęp do dowolnego tokenu, więc krótkie ścieżki gradientu i lepsze modelowanie zależności dalekiego zasięgu.
- **Residual connections + normalizacja** ułatwiają trening bardzo głębokich sieci.
- Zrównoleglenie po sekwencji pozwala wykorzystać ogromne klastry GPU i wytrenować modele na bilionach tokenów, co ujawnia emergentne zdolności.
- Dobre wsparcie transfer learningu (pre-training + fine-tuning) i skalowanie z danymi.

#### Koszty i kompromisy

- Attention ma koszt $O(n^2)$ czasu i pamięci względem długości sekwencji, a KV cache rośnie liniowo z kontekstem; RNN mają stałą pamięć i koszt $O(1)$ na token podczas inferencji.
- Stąd badania nad efektywnością: FlashAttention, sparse/sliding attention, GQA oraz architektury rekurencyjne/SSM (**Mamba**, **RWKV**) i hybrydy, które łączą efektywność RNN z jakością Transformerów.

Krótko: Transformer wymienia tańszą inferencję i mniejszą pamięć RNN na równoległy trening, lepsze zależności długiego zasięgu i skalowalność, co zdecydowało o jego dominacji.

**Źródła:**
- [Decoding Transformer Architecture (Outcome School)](https://outcomeschool.com/blog/decoding-transformer-architecture)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Wikipedia: Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)

---

<a id="q171"></a>
### 171. Wyjaśnij Query (Q), Key (K) i Value (V) w mechanizmie attention.

**Odpowiedź:**

Attention to miękkie (różniczkowalne) wyszukiwanie w słowniku. Każdy token dostaje trzy różne reprezentacje, powstałe przez rzutowanie wektora wejściowego $x_i$ trzema wyuczonymi macierzami:

$$q_i = x_i W_Q,\quad k_i = x_i W_K,\quad v_i = x_i W_V$$

#### Intuicja

- **Query** – "czego szukam?". Zapytanie tokenu, który właśnie aktualizuje swoją reprezentację.
- **Key** – "co oferuję / po czym mnie znaleźć?". Etykieta, z którą porównywane jest zapytanie.
- **Value** – "jaką informację przekażę, jeśli mnie wybierzesz?". Właściwa treść, która jest mieszana do wyjścia.

Rozdzielenie K i V jest ważne: to, po czym token jest *znajdowany*, nie musi być tym, co *przekazuje*. Np. przy rozwiązywaniu zaimka "it" klucz kodowałby "jestem rzeczownikiem w liczbie pojedynczej", a value niósłby semantykę samego rzeczownika.

#### Obliczenia

$$\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

1. $QK^\top$ – macierz $n\times n$ podobieństw (iloczyny skalarne) każdego zapytania z każdym kluczem.
2. Dzielenie przez $\sqrt{d_k}$ – iloczyny skalarne mają wariancję rosnącą z $d_k$, co spycha softmax w obszar nasycenia z bardzo małymi gradientami; skalowanie stabilizuje trening.
3. Softmax po kluczach – wagi nieujemne sumujące się do 1.
4. Mnożenie przez $V$ – ważona średnia wartości.

#### Przykład w PyTorch

```python
import torch, math
def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v
```

#### Praktyka

- Wymiary: $W_Q, W_K \in \mathbb{R}^{d_{model}\times d_k}$, $W_V \in \mathbb{R}^{d_{model}\times d_v}$ (zwykle $d_k=d_v=d_{model}/h$ w multi-head).
- W self-attention Q, K, V pochodzą z tej samej sekwencji; w cross-attention Q z dekodera, a K i V z enkodera.
- Podczas generowania K i V poprzednich tokenów są cache'owane (KV cache), a liczone jest tylko nowe Q.

**Źródła:**
- [Math behind Attention - Q, K, and V (Outcome School)](https://outcomeschool.com/blog/math-behind-attention-qkv)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al.)](https://arxiv.org/abs/1409.0473)
- [Stanford CS224n – materiały kursu](https://web.stanford.edu/class/cs224n/)

---

<a id="q172"></a>
### 172. Czym jest self-attention i jak działa w Transformerach?

**Odpowiedź:**

Self-attention (samouwaga) to mechanizm, w którym każdy token sekwencji oblicza nową reprezentację jako ważoną kombinację reprezentacji **wszystkich** tokenów tej samej sekwencji (w dekoderze: tylko poprzednich, dzięki maskowaniu). Wagi są zależne od treści – model sam uczy się, na które pozycje patrzeć.

#### Algorytm (dla jednej głowicy)

Dla macierzy wejściowej $X\in\mathbb{R}^{n\times d}$:

1. $Q=XW_Q,\;K=XW_K,\;V=XW_V$
2. $A=\text{softmax}\!\big(QK^\top/\sqrt{d_k}\big)$ – macierz $n\times n$, gdzie $A_{ij}$ mówi, ile uwagi token $i$ poświęca tokenowi $j$
3. $Z=AV$ – nowe reprezentacje
4. W pełnym bloku: residual + normalizacja + FFN.

#### Dlaczego to działa

- **Kontekstualizacja**: słowo "bank" dostaje inną reprezentację w "river bank" i "bank account".
- **Bezpośrednie połączenia**: ścieżka między dowolnymi dwoma tokenami ma długość 1 (w RNN – $O(n)$), co ułatwia propagację gradientu i uczenie zależności dalekiego zasięgu.
- **Równoległość**: cała sekwencja liczona jest jednym mnożeniem macierzy, w przeciwieństwie do sekwencyjnych RNN.

#### Koszty i ograniczenia

- Złożoność czasowa i pamięciowa $O(n^2 d)$ / $O(n^2)$ względem długości sekwencji – stąd FlashAttention, sparse/sliding-window attention, GQA.
- Self-attention jest niezmiennicze względem permutacji tokenów, więc wymaga **positional encoding** (sinusoidalne, RoPE, ALiBi).
- W dekoderze konieczna jest maska przyczynowa (causal mask), by token $i$ nie widział przyszłości.

#### Kod

```python
class SelfAttention(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.out = torch.nn.Linear(d, d)
    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y)
```

**Źródła:**
- [Self Attention in Transformers (Outcome School)](https://outcomeschool.com/blog/self-attention-in-transformers)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [PyTorch – scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)

---

<a id="q173"></a>
### 173. Czym jest Cross Attention w Transformerach?

**Odpowiedź:**

Cross-attention to wariant attention, w którym **zapytania (Q) pochodzą z jednej sekwencji, a klucze i wartości (K, V) z innej**. Pozwala to jednej reprezentacji "odpytywać" drugą.

$$\text{CrossAttn}(X,C)=\text{softmax}\!\left(\frac{(XW_Q)(CW_K)^\top}{\sqrt{d_k}}\right)(CW_V)$$

gdzie $X$ to sekwencja dekodera (np. dotychczas wygenerowane tokeny), a $C$ to "kontekst" (np. wyjście enkodera).

#### Różnice względem self-attention

| Cecha | Self-attention | Cross-attention |
|---|---|---|
| Źródło Q | ta sama sekwencja | sekwencja docelowa (dekoder) |
| Źródło K, V | ta sama sekwencja | inna sekwencja (enkoder / inna modalność) |
| Macierz uwagi | $n\times n$ | $n_{dec}\times n_{enc}$ |
| Maska przyczynowa | tak (w dekoderze) | zwykle nie (dekoder widzi cały kontekst), ewentualnie maska paddingu |

#### Zastosowania

- **Tłumaczenie maszynowe** (oryginalny Transformer, T5, BART): dekoder przy generowaniu każdego słowa patrzy na zakodowane zdanie źródłowe.
- **Modele multimodalne**: tekst odpytuje cechy obrazu (Flamingo, część architektur VLM).
- **Diffusion (Stable Diffusion)**: U-Net używa cross-attention do wstrzykiwania embeddingów promptu tekstowego do cech obrazu.
- **Whisper**: dekoder tekstowy odpytuje reprezentacje audio.
- **Perceiver / Q-Former**: mała liczba wyuczonych latentów odpytuje duże wejście, co kompresuje informację.

#### Uwagi praktyczne

- K i V z enkodera można policzyć raz i cache'ować podczas całej generacji.
- Modele "decoder-only" (GPT, Llama) nie mają cross-attention – kontekst jest po prostu wcześniejszą częścią tej samej sekwencji.
- Koszt to $O(n_{dec}\cdot n_{enc})$, więc dla długich wejść (obrazy, audio) bywa dominujący.

**Źródła:**
- [Cross Attention in Transformers (Outcome School)](https://outcomeschool.com/blog/cross-attention-in-transformers)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356)

---

<a id="q174"></a>
### 174. Czym są mechanizmy multi-head attention? Dlaczego używa się wielu głowic uwagi?

**Odpowiedź:**

Multi-head attention (MHA) uruchamia $h$ niezależnych mechanizmów attention równolegle, każdy w niższowymiarowej podprzestrzeni, a wyniki łączy.

$$\text{head}_i=\text{Attention}(XW_Q^i,XW_K^i,XW_V^i),\qquad \text{MHA}(X)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W_O$$

Zwykle $d_k=d_v=d_{model}/h$ (np. $d_{model}=4096$, $h=32$ → 128 na głowicę), więc całkowity koszt obliczeń jest zbliżony do pojedynczej głowicy o pełnym wymiarze.

#### Po co wiele głowic

- **Różne typy relacji jednocześnie**: pojedyncza macierz uwagi to jeden rozkład prawdopodobieństwa na token; z jedną głowicą model musiałby uśredniać różne zależności (składnia, koreferencja, pozycja, semantyka). Wiele głowic pozwala każdej się wyspecjalizować.
- **Różne podprzestrzenie reprezentacji**: każda głowica ma własne $W_Q,W_K,W_V$, więc patrzy na inne aspekty wektora.
- **Stabilność i redundancja**: uśrednianie wielu głowic zmniejsza wariancję; część głowic można przyciąć (pruning) bez dużej straty jakości.
- **Efektywność sprzętowa**: głowice są niezależne, więc dobrze się równoleglą na GPU.

Empirycznie obserwuje się głowice śledzące poprzedni token, dopasowujące nawiasy, realizujące "induction heads" (kopiowanie wzorców) – ale interpretacja bywa niejednoznaczna i nie każda głowica jest użyteczna.

#### Implementacja

```python
def split_heads(x, h):            # (B, T, D) -> (B, h, T, D/h)
    B, T, D = x.shape
    return x.view(B, T, h, D // h).transpose(1, 2)

def mha(x, Wq, Wk, Wv, Wo, h):
    q, k, v = (split_heads(x @ W, h) for W in (Wq, Wk, Wv))
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return y.transpose(1, 2).reshape(x.shape) @ Wo
```

#### Trade-offy

- Więcej głowic = mniejszy wymiar na głowicę; zbyt mały $d_k$ ogranicza pojemność pojedynczej głowicy.
- KV cache rośnie liniowo z liczbą głowic KV – motywacja dla MQA i GQA.

**Źródła:**
- [Multi-Head Attention in Transformers (Outcome School)](https://outcomeschool.com/blog/multi-head-attention-in-transformers)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Are Sixteen Heads Really Better than One? (Michel et al.)](https://arxiv.org/abs/1905.10650)
- [A Mathematical Framework for Transformer Circuits (Anthropic)](https://transformer-circuits.pub/2021/framework/index.html)

---

<a id="q175"></a>
### 175. W jaki sposób attention pomaga uchwycić zależności dalekiego zasięgu (long-range dependencies)?

**Odpowiedź:**

#### Problem w RNN/LSTM

W sieciach rekurencyjnych informacja z tokenu $t-k$ musi przejść przez $k$ kolejnych kroków, każdy z nieliniową transformacją. Gradient w BPTT jest iloczynem $k$ jakobianów, co prowadzi do zanikania lub eksplodowania gradientu. LSTM/GRU łagodzą problem bramkami, ale stan ukryty o stałym rozmiarze jest wąskim gardłem – cała historia musi się w nim zmieścić.

#### Rozwiązanie: bezpośredni dostęp

W self-attention token $i$ oblicza wagi dla **każdego** tokenu $j$ w jednym kroku:

- **Długość ścieżki** między dowolnymi dwoma pozycjami wynosi $O(1)$ (RNN: $O(n)$, CNN: $O(\log_k n)$ lub $O(n/k)$).
- **Gradient** płynie bezpośrednio przez wagę $\alpha_{ij}$, bez wielokrotnego mnożenia jakobianów – znacznie łatwiejsze uczenie odległych zależności.
- **Brak kompresji do stałego wektora**: wszystkie stany poprzednich tokenów (K, V) są dostępne, więc model może "wrócić" do dowolnego szczegółu.
- **Dynamiczne wagi**: połączenia zależą od treści, więc np. zaimek "ona" może silnie wskazać rzeczownik sprzed 200 tokenów.

W połączeniu z połączeniami rezydualnymi i layer norm cały sygnał ma też krótką ścieżkę przez głębokość sieci.

#### Ograniczenia

- Koszt $O(n^2)$ – ogranicza praktyczną długość kontekstu.
- W bardzo długich kontekstach softmax "rozprasza" uwagę, a modele wykazują efekt "lost in the middle" (gorzej wykorzystują informacje ze środka kontekstu).
- Pozycje muszą być kodowane (RoPE, ALiBi); ekstrapolacja poza długość treningową nie jest gwarantowana.
- Rozwiązania: sliding window/sparse attention (Longformer), FlashAttention, GQA, rozszerzanie RoPE, retrieval.

**Źródła:**
- [Decoding Transformer Architecture (Outcome School)](https://outcomeschool.com/blog/decoding-transformer-architecture)
- [Math behind Attention - Q, K, and V (Outcome School)](https://outcomeschool.com/blog/math-behind-attention-qkv)
- [Attention Is All You Need – tabela długości ścieżek (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

---

<a id="q176"></a>
### 176. Czym jest Grouped-Query Attention (GQA) i czym różni się od Multi-Head Attention (MHA)?

**Odpowiedź:**

GQA to kompromis między MHA a Multi-Query Attention (MQA), którego głównym celem jest **zmniejszenie KV cache i przepustowości pamięci podczas inferencji**.

#### Trzy warianty

| Wariant | Głowice Q | Głowice K/V | Charakterystyka |
|---|---|---|---|
| MHA | $h$ | $h$ | najlepsza jakość, największy KV cache |
| MQA | $h$ | 1 | najmniejszy cache, spadek jakości/niestabilność |
| GQA | $h$ | $g$ ($1<g<h$) | prawie jakość MHA, cache zbliżony do MQA |

W GQA $h$ głowic zapytań dzielone jest na $g$ grup; głowice w tej samej grupie współdzielą jedną parę K/V.

#### Dlaczego to ważne

Podczas dekodowania autoregresyjnego wąskim gardłem nie są FLOPs, lecz **odczyt KV cache z pamięci GPU** w każdym kroku. Rozmiar cache:

$$\text{KV bytes}=2\cdot L\cdot n_{kv}\cdot d_{head}\cdot T\cdot B\cdot \text{bytes}$$

gdzie $L$ – liczba warstw, $n_{kv}$ – liczba głowic KV, $T$ – długość kontekstu, $B$ – batch. Przy np. $h=32$ i $g=8$ cache jest 4x mniejszy niż w MHA, co pozwala na dłuższe konteksty, większe batch'e i wyższy throughput.

#### Uwagi

- Oryginalna praca pokazuje, że model MHA można **uptrainować** do GQA (średnia głowic K/V w grupie + krótki dodatkowy trening, rzędu ~5% budżetu pretrainingu).
- GQA stosowane jest m.in. w Llama 2 70B, Llama 3, Mistral 7B.
- Implementacja: `repeat_interleave` K/V do liczby głowic Q lub natywne wsparcie w `scaled_dot_product_attention` (parametr `enable_gqa` w nowszych PyTorch).

```python
k = k.repeat_interleave(h // g, dim=1)   # (B, g, T, d) -> (B, h, T, d)
v = v.repeat_interleave(h // g, dim=1)
```

**Źródła:**
- [Grouped Query Attention (Outcome School)](https://outcomeschool.com/blog/grouped-query-attention)
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150)
- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)

---

<a id="q177"></a>
### 177. Czym są sieci Feed-Forward (FFN) w LLM?

**Odpowiedź:**

FFN (position-wise feed-forward network) to drugi obok attention podstawowy podblok każdej warstwy Transformera. Jest stosowany **niezależnie do każdego tokenu** (bez mieszania informacji między pozycjami) i ma te same wagi dla wszystkich pozycji w danej warstwie.

#### Postać

Klasyczny wariant:

$$\text{FFN}(x)=W_2\,\sigma(W_1x+b_1)+b_2,\qquad W_1\in\mathbb{R}^{d\times d_{ff}},\; d_{ff}\approx 4d$$

Nowoczesne LLM (Llama, PaLM, Mistral) używają **gated FFN (SwiGLU)**:

$$\text{FFN}(x)=W_2\big(\text{SiLU}(xW_g)\odot(xW_u)\big)$$

z trzema macierzami; $d_{ff}$ zmniejsza się do ok. $\tfrac{8}{3}d$, by zachować liczbę parametrów.

#### Rola

- **Nieliniowość i pojemność**: attention jest w dużej mierze liniową kombinacją wartości; to FFN dodaje głęboką nieliniową transformację cech.
- **Pamięć wiedzy**: FFN zawiera zwykle około 2/3 parametrów modelu. Badania (Geva et al.) interpretują go jako pamięć klucz–wartość: wiersze $W_1$ wykrywają wzorce w kontekście, kolumny $W_2$ promują odpowiednie tokeny. Uważa się, że znaczna część "wiedzy faktograficznej" jest tu przechowywana.
- **Mieszanie kanałów**: attention miesza informacje *między tokenami*, FFN *między wymiarami* wektora.

#### Warianty

- Aktywacje: ReLU → GELU → SwiGLU/GeGLU (zwykle lepsza jakość przy tym samym koszcie).
- **MoE**: FFN zastępowane jest zbiorem ekspertów, z których router wybiera top-k (Mixtral, Switch).

```python
class SwiGLU(nn.Module):
    def __init__(self, d, dff):
        super().__init__()
        self.g = nn.Linear(d, dff, bias=False)
        self.u = nn.Linear(d, dff, bias=False)
        self.o = nn.Linear(dff, d, bias=False)
    def forward(self, x):
        return self.o(F.silu(self.g(x)) * self.u(x))
```

**Źródła:**
- [Feed-Forward Networks in LLMs (Outcome School)](https://outcomeschool.com/blog/feed-forward-networks-in-llms)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [GLU Variants Improve Transformer (Shazeer, 2020)](https://arxiv.org/abs/2002.05202)
- [Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al.)](https://arxiv.org/abs/2012.14913)

---

<a id="q178"></a>
### 178. Tokenizacja w dużych modelach językowych (LLM).

**Odpowiedź:**

Tokenizacja to zamiana surowego tekstu na sekwencję **tokenów** – dyskretnych jednostek ze słownika (vocabulary) o stałym rozmiarze (zwykle 32k–256k), każda mapowana na ID, a potem na embedding. Model nigdy nie widzi znaków ani słów, tylko ID tokenów.

#### Potok

1. **Normalizacja** (Unicode NFC/NFKC, opcjonalnie lowercase).
2. **Pre-tokenizacja** (podział np. po spacjach/regexie).
3. **Segmentacja podsłowna** (BPE, WordPiece, Unigram).
4. **Mapowanie na ID** + tokeny specjalne (`<bos>`, `<eos>`, `<pad>`, znaczniki czatu).

#### Poziomy tokenizacji

| Poziom | Zalety | Wady |
|---|---|---|
| Słowa | krótkie sekwencje | ogromny słownik, OOV |
| Znaki | brak OOV, mały słownik | bardzo długie sekwencje ($O(n^2)$ attention) |
| Podsłowa / bajty | kompromis, brak OOV | tokenizacja nie zawsze zgodna z morfologią |

#### Konsekwencje praktyczne

- **Koszt i limity**: zarówno cena API, jak i context window liczone są w tokenach. Angielski to ok. 0,75 słowa na token; dla polskiego i innych języków słabiej reprezentowanych w danych treningowych tokenów jest zwykle wyraźnie więcej.
- **Trudności modeli**: liczenie liter w słowie, arytmetyka, odwracanie stringów, wrażliwość na białe znaki – częściowo wynikają z tokenizacji.
- **Tokenizer jest częścią modelu**: zmiana tokenizera wymaga ponownego treningu; niezgodny tokenizer przy fine-tuningu psuje jakość.
- Trening tokenizera to osobny etap na próbce korpusu; rozmiar słownika wpływa na rozmiar warstwy embedding i softmax na wyjściu.

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
ids = tok("Tokenizacja jest ważna")["input_ids"]
print(tok.convert_ids_to_tokens(ids))
```

**Źródła:**
- [Tokenization in Large Language Models (LLMs) – Amit Shekhar (LinkedIn)](https://www.linkedin.com/posts/amit-shekhar-iitbhu_machinelearning-datascience-deeplearning-activity-7346774040784605186-ivoU)
- [Hugging Face LLM Course – Tokenizers](https://huggingface.co/learn/llm-course/chapter2/4)
- [SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)

---

<a id="q179"></a>
### 179. Czym jest tokenizacja podsłowna (subword tokenization)?

**Odpowiedź:**

Tokenizacja podsłowna dzieli tekst na jednostki pośrednie między znakami a słowami: **częste słowa pozostają całe, rzadkie są rozbijane na częste fragmenty**. Np. `unhappiness` → `un` + `happi` + `ness`.

#### Motywacja

- Słownik na poziomie słów: ogromny (setki tysięcy form, szczególnie w językach fleksyjnych jak polski) i problem **OOV** (out-of-vocabulary).
- Znaki: brak OOV, ale sekwencje 4-5x dłuższe, a pojedyncze znaki niosą mało semantyki.
- Subword: ograniczony słownik (np. 32k-100k), brak OOV (w ostateczności rozbicie do znaków/bajtów), rozsądna długość sekwencji, współdzielenie morfemów (`nauczyciel`, `nauczycielka` mają wspólne fragmenty).

#### Główne algorytmy

- **BPE** (Byte Pair Encoding): iteracyjne scalanie najczęstszej pary symboli (GPT, Llama, w wariancie byte-level).
- **WordPiece**: scalanie pary maksymalizującej wiarygodność modelu języka (BERT); kontynuacje oznaczane `##`.
- **Unigram LM**: zaczyna od dużego słownika i usuwa kandydatów o najmniejszym wpływie na likelihood; pozwala na probabilistyczne, wielokrotne segmentacje (T5, ALBERT przez SentencePiece).
- **SentencePiece**: biblioteka działająca na surowym tekście (spacja to zwykły symbol `▁`), niezależna od języka.

#### Wady i pułapki

- Segmentacja nie zawsze pokrywa się z morfologią; rzadkie słowa, nazwy własne, liczby są dzielone nieintuicyjnie.
- Języki niedoreprezentowane w korpusie tokenizera dają dłuższe sekwencje (wyższy koszt i mniej kontekstu).
- Zmiana słownika = nowy model.

```python
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("bert-base-uncased")
print(t.tokenize("unbelievably"))   # np. ['un', '##bel', '##ie', '##va', '##bly']
```

Dokładny podział zależy od słownika.

**Źródła:**
- [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al.)](https://arxiv.org/abs/1508.07909)
- [Subword Regularization: Unigram Language Model (Kudo)](https://arxiv.org/abs/1804.10959)
- [SentencePiece (Kudo & Richardson)](https://arxiv.org/abs/1808.06226)
- [Hugging Face – Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary)

---

<a id="q180"></a>
### 180. Czym jest BPE (Byte Pair Encoding) w LLM?

**Odpowiedź:**

BPE to algorytm budowy słownika podsłownego. Pierwotnie technika kompresji danych (Gage, 1994), do NLP zaadaptowana przez Sennricha i in. (2016). Zaczyna od słownika znaków (lub bajtów) i **wielokrotnie scala najczęściej występującą parę sąsiadujących symboli** w nowy symbol.

#### Trening (uczenie reguł scalania)

1. Podziel korpus na słowa (pre-tokenizacja) i rozbij je na znaki; zlicz częstości słów.
2. Policz częstości wszystkich par sąsiednich symboli.
3. Scal najczęstszą parę w nowy token, dodaj do słownika i zapamiętaj regułę merge.
4. Powtarzaj do osiągnięcia zadanego rozmiaru słownika.

**Przykład** (słowa: `low`×5, `lower`×2, `newest`×6): początkowo `l o w`, `n e w e s t`. Najczęstsza para `e s` (6) → `es`; potem `es t` → `est`; potem `l o` → `lo`, `lo w` → `low` itd. Kolejność reguł jest zapisana.

#### Kodowanie (inferencja)

Nowy tekst rozbijany jest na znaki/bajty i reguły merge stosowane są **w kolejności ich nauczenia**, aż nie da się scalić więcej.

#### Byte-level BPE

GPT-2 i nowsze operują na **bajtach UTF-8** (bazowy alfabet = 256 bajtów), więc każdy tekst (emoji, dowolny język, kod) jest reprezentowalny – brak tokenu `<unk>`. Kosztem są dłuższe sekwencje dla znaków wielobajtowych (np. polskie ą, ę mogą zajmować kilka tokenów w słabo dopasowanym słowniku).

#### Zalety i wady

- Prosty, deterministyczny, dobrze skalowalny; częste ciągi (` the`, `ing`) to pojedyncze tokeny.
- Zachłanny – nie optymalizuje bezpośrednio żadnej funkcji celu; segmentacja zależy od korpusu.
- Sensowne rozmiary słowników: ok. 32k (Llama 2), 50k (GPT-2), 128k (Llama 3), większe dla modeli wielojęzycznych.

```python
# minimalny szkic jednego kroku
from collections import Counter
def most_frequent_pair(words):        # words: dict[tuple[str,...], int]
    pairs = Counter()
    for w, f in words.items():
        for a, b in zip(w, w[1:]): pairs[(a, b)] += f
    return pairs.most_common(1)[0][0]
```

**Źródła:**
- [BPE (Byte Pair Encoding) in LLMs – Pallavi Shekhar (LinkedIn)](https://www.linkedin.com/posts/pallavi-shekhar_ai-llm-machinelearning-activity-7439218251714166784-XA4O)
- [Neural Machine Translation of Rare Words with Subword Units (Sennrich et al.)](https://arxiv.org/abs/1508.07909)
- [Hugging Face LLM Course – Byte-Pair Encoding tokenization](https://huggingface.co/learn/llm-course/chapter6/5)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

<a id="q181"></a>
### 181. Czym jest positional embedding w LLM?

**Odpowiedź:**

Self-attention jest **niezmiennicze względem permutacji** (traktuje wejście jak zbiór): bez informacji o pozycji zdania "pies gryzie człowieka" i "człowiek gryzie psa" miałyby te same reprezentacje bag-of-tokens. Positional embedding (kodowanie pozycji) wstrzykuje informację o kolejności.

#### Główne podejścia

| Metoda | Idea | Użycie |
|---|---|---|
| Sinusoidalne (absolute) | stały wektor $PE_{pos,2i}=\sin(pos/10000^{2i/d})$, $PE_{pos,2i+1}=\cos(\cdot)$ dodawany do embeddingu | oryginalny Transformer |
| Uczone (absolute) | osobny wektor uczony dla każdej pozycji $0..L_{max}$ | BERT, GPT-2 |
| Relatywne (T5 bias, Shaw) | bias zależny od odległości $i-j$ dodawany do logitów attention | T5 |
| **RoPE** | rotacja Q i K zależna od pozycji; iloczyn skalarny zależy od różnicy pozycji | Llama, Mistral, Qwen |
| **ALiBi** | liniowa kara $-m\cdot|i-j|$ dodawana do logitów | BLOOM, MPT |

#### Kluczowe właściwości

- **Ekstrapolacja długości**: uczone absolute embeddings nie mają wektorów poza $L_{max}$; sinusoidalne i uczone słabo uogólniają na dłuższe sekwencje. RoPE wymaga skalowania (Position Interpolation, NTK, YaRN), ALiBi ekstrapoluje lepiej.
- **Informacja względna** jest zwykle ważniejsza niż bezwzględna – stąd popularność RoPE.
- Positional embedding dodawany do wejścia wpływa na wszystkie warstwy przez residual; RoPE/ALiBi działają bezpośrednio w każdej warstwie attention.
- Modele bez jawnego kodowania (NoPE) potrafią uczyć się pozycji dzięki maskowaniu przyczynowemu, ale zwykle są słabsze.

```python
import torch, math
def sinusoidal(T, d):
    pos = torch.arange(T)[:, None]
    i = torch.arange(0, d, 2)[None]
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(T, d); pe[:, 0::2] = ang.sin(); pe[:, 1::2] = ang.cos()
    return pe
```

**Źródła:**
- [Understanding Positional Embedding in LLMs – Amit Shekhar (LinkedIn)](https://www.linkedin.com/posts/amit-shekhar-iitbhu_machinelearning-datascience-deeplearning-activity-7347119540482265089-cTT1)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Train Short, Test Long: Attention with Linear Biases (ALiBi)](https://arxiv.org/abs/2108.12409)

---

<a id="q182"></a>
### 182. Czym jest temperatura (temperature) w kontekście LLM?

**Odpowiedź:**

Temperatura to hiperparametr **próbkowania (sampling)** skalujący logity przed softmaxem i kontrolujący losowość generowanego tekstu.

$$p_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

gdzie $z_i$ – logit tokenu $i$, $T>0$ – temperatura.

#### Efekt

- $T\to 0$: rozkład staje się coraz ostrzejszy, w granicy to **greedy decoding** (argmax) – deterministyczny, powtarzalny.
- $T=1$: rozkład "natywny" modelu.
- $T>1$: rozkład spłaszcza się, rośnie prawdopodobieństwo rzadkich tokenów – większa kreatywność, ale i ryzyko niespójności oraz halucynacji.
- $T\to\infty$: rozkład zbliża się do jednostajnego (losowe tokeny).

**Przykład**: logity $[2,1,0]$. Dla $T=1$: $p\approx[0.67,0.24,0.09]$. Dla $T=0.5$ (logity $[4,2,0]$): $p\approx[0.87,0.12,0.02]$. Dla $T=2$ (logity $[1,0.5,0]$): $p\approx[0.51,0.31,0.19]$.

#### Praktyka

- Zadania faktograficzne, kod, ekstrakcja, klasyfikacja: niska temperatura (0-0,3).
- Kreatywne pisanie, burza mózgów: wyższa (0,7-1,2).
- Temperatura **nie zmienia rankingu tokenów**, tylko ich prawdopodobieństwa. Często łączy się ją z **top-k**, **top-p (nucleus)** lub min-p, które obcinają ogon rozkładu (Holtzman et al.).
- Nawet $T=0$ w praktyce nie zawsze daje pełną powtarzalność (niedeterminizm obliczeń zmiennoprzecinkowych na GPU, batching).
- Niska temperatura nie eliminuje halucynacji – model może z pewnością powtarzać błędny fakt.

```python
def sample(logits, T=0.7):
    probs = torch.softmax(logits / T, dim=-1)
    return torch.multinomial(probs, 1)
```

**Źródła:**
- [How does Temperature control LLM output? (Outcome School)](https://outcomeschool.com/blog/how-does-temperature-control-llm-output)
- [The Curious Case of Neural Text Degeneration (Holtzman et al.)](https://arxiv.org/abs/1904.09751)
- [Distilling the Knowledge in a Neural Network – temperatura w softmax (Hinton et al.)](https://arxiv.org/abs/1503.02531)
- [Hugging Face – Text generation strategies](https://huggingface.co/docs/transformers/generation_strategies)

---

<a id="q183"></a>
### 183. Czym jest maskowanie przyczynowe (causal masking)?

**Odpowiedź:**

Causal masking to maska w self-attention uniemożliwiająca tokenowi na pozycji $i$ "patrzenie" na tokeny $j>i$. Dzięki temu model dekoderowy modeluje rozkład autoregresyjny:

$$p(x_1,\dots,x_n)=\prod_{t=1}^{n}p(x_t\mid x_{<t})$$

#### Mechanizm

Do logitów attention dodaje się macierz trójkątną: dla $j>i$ wartość $-\infty$, dla $j\le i$ wartość 0. Po softmaxie wagi dla przyszłych pozycji wynoszą dokładnie 0.

$$A=\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right),\quad M_{ij}=\begin{cases}0&j\le i\\-\infty&j>i\end{cases}$$

#### Po co

- **Brak wycieku informacji** przy treningu: w teacher forcing cała sekwencja podawana jest naraz, a bez maski model "podglądałby" następny token i uczyłby się kopiować odpowiedź.
- **Równoległy trening**: jedna sekwencja o długości $n$ daje $n$ przykładów treningowych (predykcja każdego następnego tokenu) w jednym forward passie.
- **Spójność z inferencją**: podczas generowania przyszłość i tak nie istnieje; trening z maską odwzorowuje ten warunek, co umożliwia **KV cache** (reprezentacje wcześniejszych tokenów się nie zmieniają).

#### Porównanie

| Architektura | Maska |
|---|---|
| Decoder-only (GPT, Llama) | przyczynowa |
| Encoder (BERT) | brak (dwukierunkowa), tylko padding |
| Encoder-decoder (T5) | enkoder dwukierunkowy, dekoder przyczynowy + cross-attention |
| Prefix-LM | dwukierunkowa dla prefiksu, przyczynowa dla reszty |

Oprócz maski przyczynowej stosuje się maskę paddingu, by ignorować tokeny `<pad>`.

```python
T = q.size(-2)
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~mask, float("-inf"))
```

**Źródła:**
- [Causal Masking in Attention (Outcome School)](https://outcomeschool.com/blog/causal-masking-in-attention)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165)
- [Hugging Face – Attention mask](https://huggingface.co/docs/transformers/glossary#attention-mask)

---

<a id="q184"></a>
### 184. Czym są skip connections (połączenia rezydualne)?

**Odpowiedź:**

Skip (residual) connection dodaje wejście bloku do jego wyjścia:

$$y=x+F(x)$$

Blok $F$ uczy się więc **rezydualnej poprawki** do tożsamości, a nie pełnej transformacji. Idea pochodzi z ResNet (He et al., 2015), a w Transformerach otacza zarówno attention, jak i FFN.

#### Dlaczego to działa

- **Przepływ gradientu**: $\frac{\partial y}{\partial x}=I+\frac{\partial F}{\partial x}$. Człon $I$ zapewnia bezpośrednią ścieżkę dla gradientu do wcześniejszych warstw, co łagodzi problem zanikającego gradientu i umożliwia trening sieci z setkami warstw.
- **Łatwość uczenia tożsamości**: jeśli dodatkowa warstwa nie jest potrzebna, wystarczy $F\to 0$; głębsza sieć nie powinna być gorsza od płytszej (problem degradacji).
- **Gładsza powierzchnia funkcji straty** (Li et al., "Visualizing the Loss Landscape").
- W LLM: strumień rezydualny (residual stream) jest wspólną "szyną", do której każda warstwa dopisuje informacje; interpretowalność (circuits) traktuje go jako kanał komunikacji między głowicami i FFN.

#### Warianty w Transformerach

- **Post-LN**: $\text{LN}(x+F(x))$ – oryginał; wymaga warm-up, mniej stabilny dla bardzo głębokich modeli.
- **Pre-LN**: $x+F(\text{LN}(x))$ – stabilniejszy, standard w GPT/Llama (Xiong et al.).
- Inne: DenseNet (konkatenacja zamiast sumy), highway networks (bramkowane), U-Net (skip między enkoderem i dekoderem).

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

#### Uwagi

Wymiary $x$ i $F(x)$ muszą się zgadzać (w CNN stosuje się projekcję 1x1 przy zmianie liczby kanałów). Bez skip connections głębokie Transformery praktycznie się nie trenują (rank collapse macierzy attention).

**Źródła:**
- [What residual (skip) connections do in Transformers – Amit Shekhar (X)](https://x.com/amitiitbhu/status/2008473617806553142)
- [Deep Residual Learning for Image Recognition (He et al.)](https://arxiv.org/abs/1512.03385)
- [On Layer Normalization in the Transformer Architecture (Pre-LN)](https://arxiv.org/abs/2002.04745)
- [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913)

---

<a id="q185"></a>
### 185. Jak działa Rotary Position Embedding (RoPE) i dlaczego jest preferowane nad uczonymi positional embeddings?

**Odpowiedź:**

RoPE koduje pozycję przez **obrót** wektorów Q i K w płaszczyznach 2D, tak że iloczyn skalarny zależy wyłącznie od **względnej** odległości tokenów.

#### Mechanizm

Wektor o wymiarze $d$ dzielimy na $d/2$ par współrzędnych. Dla pozycji $m$ i pary $i$ stosujemy obrót o kąt $m\theta_i$, gdzie $\theta_i=10000^{-2i/d}$:

$$\begin{pmatrix}x_{2i}'\\x_{2i+1}'\end{pmatrix}=\begin{pmatrix}\cos m\theta_i&-\sin m\theta_i\\ \sin m\theta_i&\cos m\theta_i\end{pmatrix}\begin{pmatrix}x_{2i}\\x_{2i+1}\end{pmatrix}$$

Obrót stosuje się do $q$ (pozycja $m$) i $k$ (pozycja $n$). Ponieważ $R_m^\top R_n=R_{n-m}$:

$$\langle R_m q,\;R_n k\rangle=q^\top R_{n-m}\,k$$

czyli wynik zależy od $n-m$, a nie od pozycji bezwzględnych. Obraca się tylko Q i K (nie V), w każdej warstwie.

#### Zalety względem uczonych absolute embeddings

- **Informacja względna wbudowana w attention** – zwykle ważniejsza niż pozycja absolutna.
- **Brak dodatkowych parametrów** i brak twardego limitu $L_{max}$ wynikającego z tablicy embeddingów.
- **Lepsze uogólnienie na długość**: częstotliwości $\theta_i$ tworzą spektrum od szybkich do wolnych obrotów; z pomocą Position Interpolation, NTK-aware scaling i YaRN kontekst można rozszerzyć kilkukrotnie względem treningu (zwykle z krótkim fine-tuningiem).
- Naturalny **zanik** wpływu z odległością i zachowanie normy wektorów (obrót jest izometrią).
- Kompatybilność z KV cache i FlashAttention.

#### Wady

Ekstrapolacja poza długość treningową bez modyfikacji pogarsza jakość; wysokie wymiary o niskiej częstotliwości mogą być niedotrenowane.

```python
def rope(x, cos, sin):            # x: (B,h,T,d); cos/sin: (T, d/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
    return out.flatten(-2)
```

**Źródła:**
- [Math Behind RoPE (Rotary Position Embedding) (Outcome School)](https://outcomeschool.com/blog/math-behind-rope-rotary-position-embedding)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [Extending Context Window of LLMs via Position Interpolation](https://arxiv.org/abs/2306.15595)
- [YaRN: Efficient Context Window Extension of LLMs](https://arxiv.org/abs/2309.00071)

---

<a id="q186"></a>
### 186. Czym jest normalizacja (normalization)? Wyjaśnij RMSNorm (Root Mean Square Layer Normalization).

**Odpowiedź:**

Normalizacja w sieciach neuronowych przeskalowuje aktywacje do stabilnego zakresu (zwykle zerowa średnia, jednostkowa wariancja), co **stabilizuje i przyspiesza trening**, pozwala na większy learning rate i zmniejsza wrażliwość na inicjalizację.

#### Rodzaje

| Metoda | Po czym normalizuje | Typowe użycie |
|---|---|---|
| BatchNorm | po batchu (per kanał) | CNN; zależy od batch size, kłopotliwa dla sekwencji |
| LayerNorm | po cechach jednego tokenu | Transformery (BERT, GPT-2) |
| GroupNorm | po grupach kanałów | detekcja, dyfuzja |
| **RMSNorm** | po cechach, bez odejmowania średniej | Llama, Mistral, Gemma, Qwen |

#### LayerNorm

$$\text{LN}(x)=\gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\quad \mu=\tfrac1d\sum x_i,\;\sigma^2=\tfrac1d\sum(x_i-\mu)^2$$

#### RMSNorm

Zhang i Sennrich zauważyli, że korzyść z LayerNorm wynika głównie z **re-skalowania**, a nie z centrowania. RMSNorm pomija średnią i bias:

$$\text{RMSNorm}(x)=\gamma\odot\frac{x}{\text{RMS}(x)},\qquad \text{RMS}(x)=\sqrt{\tfrac1d\sum_{i=1}^d x_i^2+\epsilon}$$

**Przykład**: $x=[3,4]$: $\text{RMS}=\sqrt{(9+16)/2}=\sqrt{12.5}\approx3.54$, więc wynik to $\approx[0.85,1.13]$ (przed skalą $\gamma$).

#### Zalety RMSNorm

- Mniej operacji (brak liczenia średniej i $\beta$), więc szybciej – w publikacji raportowano przyspieszenie rzędu 7-64% zależnie od modelu i implementacji.
- Jakość zwykle porównywalna z LayerNorm.
- Mniej parametrów (tylko $\gamma$).
- Obliczenia zwykle w float32 dla stabilności numerycznej, nawet gdy model działa w bf16.

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.g
```

**Źródła:**
- [RMSNorm (Root Mean Square Layer Normalization) (Outcome School)](https://outcomeschool.com/blog/rmsnorm-root-mean-square-layer-normalization)
- [Root Mean Square Layer Normalization (Zhang & Sennrich)](https://arxiv.org/abs/1910.07467)
- [Layer Normalization (Ba et al.)](https://arxiv.org/abs/1607.06450)
- [Batch Normalization (Ioffe & Szegedy)](https://arxiv.org/abs/1502.03167)

---

<a id="q187"></a>
### 187. Czym jest dropout i jak jest stosowany w LLM?

**Odpowiedź:**

Dropout to technika regularyzacji: podczas treningu każdy neuron (aktywacja) jest z prawdopodobieństwem $p$ zerowany, a pozostałe skalowane. Zapobiega to **współadaptacji cech** i zmniejsza overfitting; interpretuje się go też jako trening zespołu wielu "przerzedzonych" sieci.

#### Mechanizm (inverted dropout)

Trening: $y=\dfrac{m\odot x}{1-p},\; m_i\sim\text{Bernoulli}(1-p)$. Skalowanie $1/(1-p)$ sprawia, że wartość oczekiwana wyjścia jest niezmieniona, więc **w inferencji dropout jest wyłączony** (`model.eval()`), bez dodatkowych korekt.

```python
drop = nn.Dropout(p=0.1)
model.train()   # dropout aktywny
model.eval()    # dropout wyłączony
```

#### Dropout w Transformerach

- **Attention dropout** – na macierzy wag uwagi po softmaxie.
- **Residual dropout** – na wyjściu podbloku attention/FFN przed dodaniem do strumienia rezydualnego.
- **Embedding dropout** – na sumie embeddingów tokenów i pozycji.
- Oryginalny Transformer i BERT: $p=0.1$.

#### Dropout w dużych LLM

Nowoczesne LLM pretrenowane na ogromnych korpusach w około jednej epoce **zwykle nie używają dropoutu** (lub $p=0$): overfitting nie jest głównym problemem, gdy model widzi każdy przykład rzadko, a dropout spowalnia zbieżność. Regularyzację zapewniają skala danych, weight decay i wczesne zatrzymanie. Dropout wraca w:

- **fine-tuningu na małych zbiorach** (np. LoRA dropout 0,05-0,1),
- treningu wielu epok na ograniczonych danych,
- mniejszych modelach i architekturach wizyjnych/BERT-owych.

#### Uwagi

- Wysoki $p$ przy małym modelu powoduje niedouczenie.
- Nie łączyć bezmyślnie z BatchNorm (interakcja wariancji).
- MC dropout: pozostawienie dropoutu w inferencji i uśrednianie wielu przebiegów daje przybliżoną estymację niepewności.

**Źródła:**
- [Dropout in Neural Networks (Outcome School)](https://outcomeschool.com/blog/dropout-in-neural-networks)
- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al.)](https://jmlr.org/papers/v15/srivastava14a.html)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Dropout as a Bayesian Approximation (Gal & Ghahramani)](https://arxiv.org/abs/1506.02142)
- [PyTorch – torch.nn.Dropout](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html)

---

<a id="q188"></a>
### 188. Dlaczego Attention używa Softmax?

**Odpowiedź:**

Softmax zamienia surowe wyniki dopasowania (logity) $s_{ij}=q_i\cdot k_j/\sqrt{d_k}$ na rozkład prawdopodobieństwa:

$$\alpha_{ij}=\frac{e^{s_{ij}}}{\sum_{l}e^{s_{il}}}$$

#### Powody

1. **Normalizacja**: wagi są nieujemne i sumują się do 1, więc wyjście $\sum_j\alpha_{ij}v_j$ to **wypukła kombinacja** wartości – skala wyjścia jest stabilna niezależnie od długości sekwencji.
2. **Różniczkowalność**: gładka, więc gradient płynie do Q i K. Twardy argmax (hard attention) nie jest różniczkowalny.
3. **Selektywność ("soft argmax")**: wykładnik wzmacnia różnice – większy wynik dostaje nieproporcjonalnie większą wagę, co pozwala skupić się na kilku tokenach, a nadal utrzymuje gradient dla pozostałych. Temperaturą jest tu $\sqrt{d_k}$.
4. **Konkurencja między tokenami**: wagi sumują się do 1, więc zwiększenie uwagi na jeden token zmniejsza ją na innych, co daje interpretowalny "budżet uwagi".
5. **Dodatniość** i naturalna obsługa masek: $-\infty$ → waga 0.
6. **Interpretacja probabilistyczna** i związek z modelem log-liniowym (Boltzmann).

#### Wady

- Koszt $O(n^2)$ i wymóg globalnej normalizacji po wierszu (utrudnia liniowe attention; FlashAttention rozwiązuje to online softmaxem).
- Wagi nie mogą być zero (zawsze coś "wycieka"), a suma =1 wymusza rozdanie uwagi nawet, gdy nic nie pasuje – stąd zjawisko **attention sinks** (nadmierna uwaga na pierwszy token) i propozycje typu "softmax-off-by-one".
- Nasycenie: duże logity → gradienty bliskie 0; dlatego dzielimy przez $\sqrt{d_k}$.

#### Alternatywy

Sigmoid attention, ReLU/ReLU$^2$ attention, sparsemax, liniowe/kernelowe attention (kosztem jakości, choć niektóre prace pokazują konkurencyjne wyniki).

```python
weights = torch.softmax(scores, dim=-1)   # po osi kluczy
```

**Źródła:**
- [Why does Attention use Softmax? – Amit Shekhar (X)](https://x.com/amitiitbhu/status/2005495879571255670)
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)
- [Online normalizer calculation for softmax (Milakov & Gimelshein)](https://arxiv.org/abs/1805.02867)

---

<a id="q189"></a>
### 189. Co przechowuje baza wektorowa (Vector DB) w zastosowaniach LLM?

**Odpowiedź:**

Baza wektorowa przechowuje **embeddingi** – gęste wektory liczbowe (zwykle 384-3072 wymiarów) reprezentujące semantykę tekstu, obrazów lub audio – oraz umożliwia szybkie wyszukiwanie **najbliższych sąsiadów** (nearest neighbors).

#### Co jest w rekordzie

- **ID** dokumentu/fragmentu (chunk).
- **Wektor** (embedding) wygenerowany przez model embeddingowy.
- **Payload / metadane**: źródło, tytuł, data, autor, uprawnienia, numer strony, język – używane do filtrowania.
- Często **oryginalny tekst chunka** (lub referencja do niego w innym magazynie).

#### Jak jest używana (RAG)

1. **Indeksowanie**: dokumenty → chunking → embedding → zapis do bazy.
2. **Zapytanie**: pytanie → ten sam model embeddingowy → wektor zapytania.
3. **Wyszukiwanie top-k** po podobieństwie (cosine, dot product, L2).
4. **Augmentacja**: znalezione chunki trafiają do promptu LLM, który generuje odpowiedź z cytowaniem źródeł.

#### Indeksy ANN

Dokładne przeszukiwanie to $O(N\cdot d)$, więc stosuje się przybliżone (Approximate Nearest Neighbor):

- **HNSW** (graf hierarchiczny) – szybkie i dokładne, duży narzut pamięci,
- **IVF** (klastry) + **PQ** (kwantyzacja produktowa) – kompresja pamięci,
- **ScaNN, DiskANN** – skala miliardów wektorów.

Kompromis: recall vs latencja vs pamięć.

#### Inne zastosowania w LLM

Pamięć długoterminowa agentów, semantic cache odpowiedzi, deduplikacja, rekomendacje, wyszukiwanie multimodalne.

#### Praktyka

- Jakość zależy głównie od modelu embeddingowego i strategii chunkingu, a nie od samej bazy.
- Wyszukiwanie hybrydowe (wektorowe + BM25) i reranking zwykle poprawiają trafność.
- Zmiana modelu embeddingowego wymaga reindeksacji.
- Przykłady: FAISS (biblioteka), pgvector, Qdrant, Milvus, Weaviate, Pinecone, Chroma.

```python
import faiss, numpy as np
index = faiss.IndexFlatIP(384)
index.add(np.asarray(vectors, dtype="float32"))
scores, ids = index.search(query_vec, k=5)
```

**Źródła:**
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [Efficient and robust ANN search using HNSW graphs (Malkov & Yashunin)](https://arxiv.org/abs/1603.09320)
- [FAISS – dokumentacja (GitHub wiki)](https://github.com/facebookresearch/faiss/wiki)
- [The Faiss library (Douze et al.)](https://arxiv.org/abs/2401.08281)

---

<a id="q190"></a>
### 190. Jak poprawić szybkość inferencji w produkcyjnych wdrożeniach LLM?

**Odpowiedź:**

Inferencja LLM ma dwie fazy: **prefill** (przetwarzanie promptu, równoległe, ograniczone obliczeniami – compute-bound) i **decode** (generowanie token po tokenie, ograniczone przepustowością pamięci – memory-bound, bo w każdym kroku czytane są wszystkie wagi i KV cache). Metryki: **TTFT** (time to first token), **TPOT/ITL** (czas na token), throughput (tokeny/s), koszt/token.

#### Techniki (od najbardziej opłacalnych)

**1. Kwantyzacja** – wagi w INT8/INT4 (GPTQ, AWQ) lub FP8; mniejszy ruch pamięci i mniejsze zapotrzebowanie na VRAM. Również kwantyzacja KV cache. Weryfikuj spadek jakości na własnych zadaniach.

**2. Efektywne serwowanie**
- **Continuous batching** – dokładanie nowych żądań do batcha w trakcie generacji (zamiast statycznego batchowania).
- **PagedAttention (vLLM)** – stronicowanie KV cache eliminuje fragmentację i pozwala na większe batch'e.
- Frameworki: vLLM, TensorRT-LLM, SGLang, TGI.

**3. Optymalizacje attention/pamięci**
- **FlashAttention** (kernel IO-aware), **GQA/MQA** (mniejszy KV cache), sliding window.
- **Prefix caching** – ponowne użycie KV wspólnego prefiksu (system prompt, RAG).

**4. Speculative decoding** – mały model draft proponuje kilka tokenów, duży weryfikuje je jednym przebiegiem; matematycznie zachowuje rozkład wyjściowy, zwykle przyspieszenie 2-3x. Warianty: Medusa, EAGLE.

**5. Równoległość i sprzęt** – tensor parallelism, dobór GPU (przepustowość HBM), disaggregated prefill/decode.

**6. Poziom aplikacji** – krótsze prompty, streaming odpowiedzi, limity `max_tokens`, mniejszy model/distillation dla prostych zadań (routing), cache odpowiedzi, structured output.

#### Proces

Najpierw zmierz (profilowanie, percentyle p50/p95/p99, obciążenie zbliżone do produkcyjnego), zidentyfikuj wąskie gardło (TTFT vs TPOT vs kolejkowanie) i dopiero wtedy optymalizuj. Kompromis: latencja pojedynczego żądania vs throughput całego systemu.

**Źródła:**
- [LLM Inference Optimization – Amit Shekhar (X)](https://x.com/amitiitbhu/status/2054100147546837154)
- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)

---

<a id="q191"></a>
### 191. Czym jest okno kontekstowe (context window) w LLM?

**Odpowiedź:**

Context window to **maksymalna liczba tokenów**, które model może jednocześnie uwzględnić w jednym przebiegu – obejmuje prompt (instrukcje systemowe, historię rozmowy, dokumenty RAG, definicje narzędzi) **oraz** generowaną odpowiedź. Można je traktować jak pamięć roboczą modelu: wszystko, co poza oknem, dla modelu nie istnieje (model nie ma pamięci między wywołaniami poza tym, co mu się poda).

#### Skala

Od 2-4k tokenów (wczesne GPT-3/Llama 1) przez 8k-128k (Llama 3.x, GPT-4-class) po setki tysięcy i miliony tokenów w najnowszych modelach – konkretne wartości zależą od modelu i szybko się zmieniają, więc sprawdzaj dokumentację dostawcy.

#### Konsekwencje praktyczne

- **Koszt i latencja** rosną z długością kontekstu (prefill, KV cache, cena za token wejściowy).
- **Jakość ≠ nominalna długość**: modele gorzej wykorzystują informacje ze środka długiego kontekstu ("lost in the middle"), a efektywna długość bywa krótsza od deklarowanej.
- **Przepełnienie**: gdy rozmowa przekracza okno, trzeba ją obciąć, streścić lub użyć retrievalu.
- Kontekst jest **jedynym kanałem** in-context learning (few-shot) i przekazywania świeżych danych.

#### Strategie zarządzania

- **RAG** – wstrzykuj tylko najbardziej relewantne fragmenty.
- **Streszczanie / kompaktowanie** historii, sliding window, pamięć zewnętrzna dla agentów.
- **Chunking** długich dokumentów, map-reduce.
- **Prompt caching** dla powtarzalnych prefiksów.
- Umieszczanie kluczowych informacji na początku i końcu promptu.

#### Rozszerzanie okna

Techniki skalowania RoPE (Position Interpolation, YaRN), trening na dłuższych sekwencjach, sparse/sliding attention, FlashAttention i GQA obniżające koszt.

**Źródła:**
- [Context window in LLM – Amit Shekhar (LinkedIn)](https://www.linkedin.com/posts/amit-shekhar-iitbhu_the-context-window-is-the-llms-working-memory-activity-7437754426175672320-MH9c/)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Extending Context Window of LLMs via Position Interpolation](https://arxiv.org/abs/2306.15595)
- [Hugging Face – Glossary: context window / tokens](https://huggingface.co/docs/transformers/glossary)

---

<a id="q192"></a>
### 192. Dlaczego okno kontekstowe jest ograniczone w LLM?

**Odpowiedź:**

Ograniczenie wynika z kilku niezależnych czynników – obliczeniowych, pamięciowych i związanych z treningiem.

#### 1. Kwadratowy koszt attention

Macierz uwagi ma rozmiar $n\times n$, więc obliczenia rosną jak $O(n^2 d)$. Dla kontekstu 10x dłuższego to ok. 100x więcej pracy w warstwach attention (FFN rośnie liniowo). FlashAttention usuwa konieczność materializacji macierzy $n\times n$ w pamięci (pamięć liniowa), ale **liczba FLOPs pozostaje kwadratowa**.

#### 2. Pamięć: KV cache

Podczas generacji przechowuje się K i V wszystkich poprzednich tokenów:

$$\text{KV}=2\cdot L\cdot n_{kv}\cdot d_{head}\cdot n\cdot B\cdot\text{bytes}$$

Rośnie liniowo z $n$ i z liczbą równoległych użytkowników. Dla długich kontekstów cache potrafi przewyższyć rozmiar samych wag i wyczerpać VRAM. Każdy krok dekodowania musi ten cache odczytać, więc rośnie też latencja.

#### 3. Trening na krótkich sekwencjach

Pretraining na długich sekwencjach jest drogi (koszt attention, pamięć aktywacji), więc modele trenuje się głównie na krótszych, a kontekst rozszerza w krótkim etapie końcowym. Positional encodings (np. RoPE) słabo ekstrapolują poza widziane długości – model "nie wie", co robić na pozycjach, których nigdy nie widział.

#### 4. Niedobór długich danych

Naturalnych dokumentów o wielu setkach tysięcy tokenów, w których odległe fragmenty naprawdę są ze sobą powiązane, jest mało; trzeba je syntezować.

#### 5. Jakość

Nawet gdy dłuższe okno jest technicznie możliwe, uwaga rozprasza się na wiele tokenów, a model gorzej wykorzystuje informację ze środka kontekstu.

#### Jak się to łagodzi

Sparse / sliding window attention, GQA/MQA, kwantyzacja KV cache, PagedAttention, RoPE scaling (PI, YaRN), architektury alternatywne (SSM/Mamba, hybrydy), retrieval i pamięć zewnętrzna.

**Źródła:**
- [Why is the context window limited in LLMs? (YouTube, Outcome School)](https://www.youtube.com/watch?v=CGIhxIaOg3M&lc)
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [YaRN: Efficient Context Window Extension of LLMs](https://arxiv.org/abs/2309.00071)

---

<a id="q193"></a>
### 193. Wyjaśnij Prompting, Retrieval-Augmented Generation (RAG) i Fine-Tuning.

**Odpowiedź:**

To trzy główne sposoby dopasowania LLM do zadania, różniące się tym, **co się zmienia**: prompt, kontekst czy wagi.

#### Prompting

Zmieniamy tylko wejście: instrukcje, przykłady (few-shot), format wyjścia, rozumowanie krok po kroku (chain-of-thought), rola systemowa.
- **Plusy**: najszybsze, najtańsze, zero treningu, łatwa iteracja.
- **Minusy**: ograniczone oknem kontekstowym, wiedza modelu pozostaje statyczna, wrażliwość na sformułowania, brak gwarancji stylu/formatu.

#### RAG

Przed generacją system **wyszukuje** relewantne fragmenty w zewnętrznej bazie (zwykle wektorowej + BM25) i dołącza je do promptu.
- **Plusy**: aktualna i prywatna wiedza bez retreningu, cytowanie źródeł, mniej halucynacji, łatwa aktualizacja i kontrola dostępu.
- **Minusy**: dodatkowa infrastruktura i latencja, jakość zależy od retrievalu (chunking, embeddingi, reranking); model nadal może zignorować kontekst.

#### Fine-tuning

Aktualizujemy **wagi** modelu na własnych danych (SFT, LoRA/QLoRA, preference tuning).
- **Plusy**: nauka stylu, formatu, terminologii domenowej, zachowania, użycia narzędzi; krótsze prompty i mniejszy, tańszy model o dobrej jakości w wąskiej domenie.
- **Minusy**: koszt danych i treningu, ryzyko catastrophic forgetting i overfittingu, wiedza się starzeje, trudniejsze wdrażanie/wersjonowanie; słabo nadaje się do wstrzykiwania dużej ilości faktów.

#### Kiedy co

| Potrzeba | Wybór |
|---|---|
| Szybki prototyp, proste zadania | Prompting |
| Wiedza aktualna / firmowa, cytowania | RAG |
| Spójny styl, format, zachowanie, redukcja kosztu | Fine-tuning |
| Złożone systemy | Kombinacja (np. fine-tuned model + RAG + dobry prompt) |

Zasada: zacznij od promptingu, dodaj RAG, gdy brakuje wiedzy, a fine-tuning, gdy brakuje **zachowania** lub potrzebna jest optymalizacja kosztu/latencji. Zawsze mierz zestawem ewaluacyjnym.

**Źródła:**
- [Prompting, RAG, and Fine-Tuning (Outcome School)](https://outcomeschool.substack.com/p/prompting-vs-rag-vs-fine-tuning)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903)

---

<a id="q194"></a>
### 194. Czym jest Mixture of Experts (MoE) i jak działa w modelach takich jak Mixtral?

**Odpowiedź:**

MoE zastępuje pojedynczy gęsty blok FFN w Transformerze zbiorem **ekspertów** (osobnych FFN) i **routerem**, który dla każdego tokenu wybiera tylko kilku z nich. Dzięki temu liczba parametrów rośnie, a koszt obliczeń na token pozostaje mały (**warunkowe obliczenia**, conditional computation).

#### Mechanizm

Dla tokenu $x$ router (warstwa liniowa) liczy wyniki $g=\text{softmax}(xW_r)$ i wybiera **top-k** ekspertów:

$$y=\sum_{i\in\text{TopK}(g)}\tilde g_i\,E_i(x)$$

gdzie $\tilde g_i$ to renormalizowane wagi wybranych ekspertów.

#### Mixtral 8x7B

- W każdej warstwie 8 ekspertów FFN, **top-2** na token.
- Attention jest współdzielone (gęste) – nie jest częścią MoE.
- Łącznie ok. 47B parametrów, ale aktywnych na token ok. 13B; koszt inferencji zbliżony do modelu ~13B, a jakość porównywalna z znacznie większymi gęstymi modelami (według autorów przewyższał Llama 2 70B na wielu benchmarkach).
- Różni tokeny w tej samej sekwencji mogą trafiać do różnych ekspertów; eksperci nie specjalizują się w sposób prosty tematycznie, obserwuje się raczej wzorce składniowe/pozycyjne.

#### Wyzwania

- **Load balancing**: bez regularyzacji router faworyzuje kilku ekspertów ("expert collapse"); dodaje się auxiliary loss równoważący obciążenie, capacity factor, ewentualnie strategie bez aux loss.
- **Pamięć**: wszystkie parametry muszą być w VRAM, mimo że aktywna jest ich część – zysk dotyczy FLOPs, nie pamięci.
- **Komunikacja**: expert parallelism (all-to-all) komplikuje trening i serwowanie.
- **Stabilność treningu** i trudniejszy fine-tuning (skłonność do overfittingu).

#### Warianty

Switch Transformer (top-1), GShard, DeepSeekMoE (drobnoziarniści + współdzieleni eksperci), Qwen-MoE, Grok.

```python
def moe(x, router, experts, k=2):          # x: (tokens, d)
    w, idx = torch.topk(torch.softmax(router(x), -1), k)   # (tokens, k)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros_like(x)
    for e, expert in enumerate(experts):
        tok, slot = (idx == e).nonzero(as_tuple=True)
        if tok.numel():
            out[tok] += w[tok, slot, None] * expert(x[tok])
    return out
```

Wydajne implementacje grupują tokeny per ekspert i używają dedykowanych kerneli.

**Źródła:**
- [Mixture of Experts Explained (Outcome School)](https://outcomeschool.com/blog/mixture-of-experts)
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088)
- [Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer](https://arxiv.org/abs/1701.06538)
- [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [Hugging Face Blog – Mixture of Experts Explained](https://huggingface.co/blog/moe)

---

<a id="q195"></a>
### 195. Jaka jest różnica między modelami gęstymi (dense) a rzadkimi (sparse)?

**Odpowiedź:**

#### Model gęsty (dense)

Przy każdym tokenie **aktywowane są wszystkie parametry**. Koszt obliczeń na token jest proporcjonalny do liczby parametrów (w przybliżeniu $\approx 2N$ FLOPs na token w forward passie). Przykłady: GPT-3, Llama 2/3, BERT.

#### Model rzadki (sparse)

Przy każdym tokenie aktywowany jest **tylko podzbiór parametrów** – najczęściej dzięki architekturze **Mixture of Experts**. Rozróżnia się:
- **parametry całkowite** (decydują o pojemności i zapotrzebowaniu na pamięć),
- **parametry aktywne** (decydują o FLOPs i latencji na token).

Np. Mixtral 8x7B: ok. 47B całkowitych, ok. 13B aktywnych.

#### Porównanie

| Cecha | Dense | Sparse (MoE) |
|---|---|---|
| Aktywne parametry / token | wszystkie | ułamek (np. top-2 z 8) |
| Koszt obliczeń / token | proporcjonalny do $N$ | proporcjonalny do $N_{aktywne}$ |
| Pamięć (VRAM) | $N$ | $N_{całkowite}$ (duża) |
| Jakość przy tym samym koszcie obliczeń | niższa | zwykle wyższa |
| Trening / serwowanie | proste | load balancing, expert parallelism |
| Fine-tuning | stabilny | trudniejszy, ryzyko overfittingu |
| Przepustowość dla małych batchy | dobra | mniej efektywna (wiele ekspertów ładowanych z pamięci) |

#### Inne znaczenie "sparse"

- **Rzadkość wag** (pruning, sparsity 2:4 na NVIDIA) – zerowanie wag w istniejącym modelu.
- **Rzadkie attention** (sliding window, block-sparse) – ograniczenie liczby par tokenów.
Warto zaznaczyć, w jakim sensie mowa o rzadkości.

#### Kiedy co

Sparse MoE: gdy liczy się jakość na jednostkę obliczeń i dużo VRAM/ruchu jest dostępne (serwowanie w dużej skali). Dense: prostsze wdrożenia, urządzenia brzegowe o ograniczonej pamięci, łatwiejszy fine-tuning.

**Źródła:**
- [Mixture of Experts Explained (Outcome School)](https://outcomeschool.com/blog/mixture-of-experts)
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088)
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961)
- [Hugging Face Blog – Mixture of Experts Explained](https://huggingface.co/blog/moe)

---

<a id="q196"></a>
### 196. Transformery pracują na tekście – czy potrafią też rozumieć obrazy?

**Odpowiedź:**

Tak. Transformer operuje na **sekwencji wektorów**, a nie na tekście per se – wystarczy zamienić obraz na sekwencję tokenów. Robi to **Vision Transformer (ViT)**.

#### ViT krok po kroku

1. **Patchify**: obraz $H\times W\times C$ dzielony jest na patche $P\times P$ (np. 16x16), co daje $N=HW/P^2$ patchy (dla 224x224 i $P=16$: 196).
2. **Linear projection**: każdy spłaszczony patch ($P^2C$) rzutowany jest warstwą liniową (równoważnie konwolucją z krokiem $P$) na embedding wymiaru $d$.
3. **Token [CLS]** dołączany na początek (lub global average pooling).
4. **Positional embeddings** (uczone 1D lub 2D) dodawane do patchy.
5. Standardowy **enkoder Transformera** (self-attention + FFN).
6. **Głowica klasyfikacyjna** na wyjściu tokenu [CLS].

Każdy patch to "słowo", obraz to "zdanie".

#### ViT vs CNN

- Brak wbudowanych **indukcyjnych biasów** CNN (lokalność, translation equivariance) – ViT potrzebuje **dużo danych** (JFT-300M, ImageNet-21k) lub silnej augmentacji/destylacji (DeiT), by dorównać CNN; na dużych zbiorach zwykle je przewyższa.
- Globalne attention od pierwszej warstwy; koszt $O(N^2)$ względem liczby patchy (stąd Swin Transformer z oknami).

#### Modele multimodalne

- **CLIP**: enkoder obrazu (ViT) + enkoder tekstu trenowane kontrastowo.
- **VLM (LLaVA, GPT-4V-class, Gemini, Claude)**: enkoder wizyjny → projekcja/adapter → tokeny obrazu wstawiane do sekwencji LLM (albo cross-attention).
- **Video, audio (Whisper)**: analogicznie – patche czasoprzestrzenne / spektrogramy.
- **Generatywne**: DiT (Diffusion Transformer).

```python
patches = nn.Conv2d(3, d, kernel_size=16, stride=16)(img)   # (B,d,14,14)
tokens = patches.flatten(2).transpose(1, 2)                  # (B,196,d)
```

**Źródła:**
- [Decoding Vision Transformer (ViT) (Outcome School)](https://outcomeschool.com/blog/decoding-vision-transformer-vit)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Swin Transformer](https://arxiv.org/abs/2103.14030)

---

<a id="q197"></a>
### 197. Małe modele językowe (Small Language Models, SLM).

**Odpowiedź:**

SLM to modele językowe o stosunkowo małej liczbie parametrów – zwykle od setek milionów do kilkunastu miliardów (granica jest płynna) – projektowane tak, by działały tanio, szybko i lokalnie (laptop, telefon, edge), przy zachowaniu dobrej jakości w wybranych zadaniach.

#### Przykłady

Phi-3/Phi-4 (Microsoft), Gemma (Google), Llama 3.2 1B/3B, Qwen 2.5 małe warianty, SmolLM, Mistral 7B.

#### Jak osiąga się jakość mimo małego rozmiaru

- **Wysokiej jakości dane**: kuratorowane i syntetyczne ("textbooks are all you need" – seria Phi).
- **Overtraining względem Chinchilli**: trening na dużo większej liczbie tokenów niż optimum obliczeniowe (wysoki koszt treningu, ale tani i dobry model do inferencji).
- **Destylacja wiedzy** z większego modelu (teacher → student).
- **Kwantyzacja** (INT4/INT8) i pruning.
- Efektywne architektury: GQA, wspólne embeddingi wejścia i wyjścia.
- **Fine-tuning** (LoRA) do wąskiej domeny.

#### Zalety

- Niski koszt i latencja, mały footprint pamięci.
- **Prywatność** i działanie offline (on-device), brak zależności od API.
- Łatwy fine-tuning i wdrażanie, niższy ślad energetyczny.
- W wąskich zadaniach (klasyfikacja, ekstrakcja, routing, proste agenty) często wystarczające.

#### Ograniczenia

- Słabsza wiedza ogólna, rozumowanie wieloetapowe i wielojęzyczność; mniejsze okno kontekstowe; więcej halucynacji bez retrievalu.
- Wrażliwość na prompt i dane.

#### Wzorce użycia

Routing (SLM dla prostych zapytań, duży model dla trudnych), SLM + RAG, SLM jako komponent agenta lub guardrail, speculative decoding (SLM jako model draft).

**Źródła:**
- [Small Language Models (SLMs) (Outcome School)](https://outcomeschool.com/blog/small-language-models-slms)
- [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

---

<a id="q198"></a>
### 198. Duże modele rozumujące (Large Reasoning Models, LRM).

**Odpowiedź:**

LRM to LLM wytrenowane (głównie przez uczenie ze wzmocnieniem), by przed udzieleniem odpowiedzi generować **długi, wewnętrzny łańcuch rozumowania** (chain of thought): planować, sprawdzać się, cofać i poprawiać błędy. Przykłady: OpenAI o1/o3, DeepSeek-R1, modele z "extended thinking" (Claude), Gemini Thinking, Qwen QwQ.

#### Czym różnią się od zwykłych LLM

- Standardowy LLM odpowiada niemal od razu (ewentualnie z CoT wymuszonym promptem). LRM **alokuje więcej obliczeń w czasie inferencji** (test-time compute) na trudniejsze problemy.
- Rozumowanie jest **wyuczone**, a nie tylko podpowiadane: zachowania typu weryfikacja, alternatywne podejścia, "wait, let me re-check" pojawiają się dzięki RL.

#### Jak są trenowane (w uproszczeniu)

1. Bazowy model + opcjonalnie SFT na przykładach rozumowania (cold start).
2. **RL z weryfikowalnymi nagrodami (RLVR)**: matematyka, kod, logika – nagroda z automatycznej weryfikacji poprawności (testy jednostkowe, porównanie odpowiedzi), często algorytmem GRPO/PPO. DeepSeek-R1-Zero pokazał, że rozumowanie może wyłonić się z samego RL.
3. Rozbudowa: rejection sampling, destylacja rozumowania do mniejszych modeli, process reward models (nagroda za kroki).

#### Skalowanie w czasie testu

Jakość rośnie wraz z długością rozumowania i liczbą próbek: majority voting/self-consistency, best-of-N z weryfikatorem, przeszukiwanie drzewa. Uzupełnia to skalowanie danych i parametrów w treningu.

#### Zalety i wady

- **Plusy**: znaczna poprawa w matematyce, kodowaniu, planowaniu i zadaniach naukowych wieloetapowych.
- **Minusy**: wyższy koszt i latencja (tysiące tokenów "myślenia"), zbędne "przemyślanie" prostych zapytań (overthinking), łańcuch rozumowania nie zawsze wiernie odzwierciedla rzeczywisty proces modelu, ograniczenia przy zadaniach bez weryfikowalnej nagrody, wciąż halucynacje.

#### Praktyka

Używaj LRM do trudnych zadań analitycznych, a do prostych – szybszych modeli; kontroluj budżet myślenia (parametry typu reasoning effort / thinking budget); nie stosuj rozbudowanych promptów CoT, które mogą przeszkadzać.

**Źródła:**
- [Large Reasoning Models (Outcome School)](https://outcomeschool.com/blog/large-reasoning-models)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
- [Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903)

---

<a id="q199"></a>
### 199. Uczenie ze wzmocnieniem z ludzkiej informacji zwrotnej (RLHF).

**Odpowiedź:**

RLHF to metoda **dopasowania (alignment)** modelu językowego do ludzkich preferencji – pomocności, bezpieczeństwa, stylu – w sytuacjach, gdy trudno zapisać funkcję celu wprost. Kluczowa metoda za InstructGPT i wczesnym ChatGPT.

#### Trzy etapy (Ouyang et al., 2022)

1. **SFT (Supervised Fine-Tuning)**: fine-tuning modelu bazowego na demonstracjach wysokiej jakości (instrukcja → dobra odpowiedź), co daje politykę początkową $\pi_{SFT}$.
2. **Model nagrody (Reward Model)**: dla promptu $x$ model generuje kilka odpowiedzi, ludzie je **rankingują**; z par preferencji (zwycięzca $y_w$, przegrany $y_l$) uczy się $r_\phi(x,y)$ według modelu Bradleya-Terry'ego:

$$\mathcal L_{RM}=-\mathbb E\big[\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)\big]$$

3. **Optymalizacja polityki RL (zwykle PPO)**: maksymalizacja nagrody z karą KL względem modelu referencyjnego:

$$\max_{\pi_\theta}\;\mathbb E_{x,\,y\sim\pi_\theta}\big[r_\phi(x,y)\big]-\beta\,\mathrm{KL}\big(\pi_\theta\,\|\,\pi_{ref}\big)$$

Kara KL nie pozwala polityce oddalić się zbyt daleko od SFT, co ogranicza **reward hacking** i degradację języka.

#### Wyzwania

- Koszt i jakość anotacji; szum i sprzeczność preferencji różnych anotatorów.
- **Reward hacking / overoptimization**: model wykorzystuje luki w modelu nagrody (np. rozwlekłość, schlebianie – sycophancy).
- Złożoność i niestabilność RL (cztery modele w pamięci: polityka, referencja, reward, value/critic).
- Nagroda odzwierciedla preferencje anotatorów, nie "prawdę".

#### Alternatywy

- **DPO** – optymalizuje preferencje bezpośrednio, bez osobnego reward modelu i RL.
- **RLAIF / Constitutional AI** – preferencje generowane przez model AI według zasad.
- **GRPO, RLVR** – RL z nagrodami weryfikowalnymi (rozumowanie).

**Źródła:**
- [Reinforcement Learning from Human Feedback (RLHF) (Outcome School)](https://outcomeschool.com/blog/reinforcement-learning-from-human-feedback-rlhf)
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325)
- [Deep RL from Human Preferences (Christiano et al.)](https://arxiv.org/abs/1706.03741)
- [Hugging Face Blog – Illustrating RLHF](https://huggingface.co/blog/rlhf)

---

<a id="q200"></a>
### 200. Proximal Policy Optimization (PPO).

**Odpowiedź:**

PPO (Schulman et al., 2017) to algorytm RL z rodziny **policy gradient** (actor-critic), który stabilizuje aktualizacje polityki, ograniczając, jak bardzo nowa polityka może się różnić od poprzedniej – prostsza alternatywa dla TRPO. Standardowy optymalizator w klasycznym RLHF.

#### Problem, który rozwiązuje

Zwykły policy gradient jest wrażliwy na krok: zbyt duża aktualizacja psuje politykę i trening się załamuje, a próbki są zużywane po jednej aktualizacji. PPO pozwala na **kilka epok minibatchy** na tych samych danych bezpiecznie.

#### Funkcja celu

Stosunek prawdopodobieństw: $r_t(\theta)=\dfrac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$. **Clipped objective**:

$$L^{CLIP}(\theta)=\mathbb E_t\Big[\min\big(r_t\hat A_t,\;\text{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t\big)\Big],\quad\epsilon\approx0.1\text{-}0.2$$

- Gdy $\hat A_t>0$ (dobra akcja), zysk jest obcięty powyżej $1+\epsilon$ – brak zachęty do zwiększania jej prawdopodobieństwa ponad limit.
- Gdy $\hat A_t<0$, obcięcie poniżej $1-\epsilon$.
- $\min$ daje pesymistyczną (dolną) granicę.

Całkowita strata: $L=-L^{CLIP}+c_1L^{VF}-c_2 S[\pi_\theta]$ (strata krytyka + bonus entropii).

**Advantage** $\hat A_t$ estymowany jest zwykle przez **GAE**: $\hat A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$, $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$.

#### PPO w RLHF

- "Stan" = prompt + dotychczasowe tokeny, "akcja" = następny token, nagroda z reward modelu na końcu sekwencji plus **kara KL per token** względem $\pi_{ref}$.
- Wymaga czterech modeli: polityka (actor), referencja, reward model i **krytyk** (value model) – kosztowne pamięciowo.

#### Wady

Złożoność, wrażliwość na hiperparametry i szczegóły implementacji, duże zużycie pamięci przy LLM, niestabilność. To motywacja dla DPO (bez RL) i GRPO (bez krytyka).

```python
ratio = (logp_new - logp_old).exp()
loss = -torch.min(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).mean()
```

**Źródła:**
- [Proximal Policy Optimization (PPO) (Outcome School)](https://outcomeschool.com/blog/proximal-policy-optimization-ppo)
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- [OpenAI Spinning Up – PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

---

<a id="q201"></a>
### 201. Direct Preference Optimization (DPO).

**Odpowiedź:**

DPO (Rafailov et al., 2023) to metoda dopasowania modelu do preferencji, która **eliminuje osobny model nagrody i pętlę RL**: polityka jest optymalizowana bezpośrednio prostą stratą klasyfikacyjną na parach preferencji.

#### Idea

W RLHF optymalna polityka dla nagrody $r$ z karą KL ma postać $\pi^*(y|x)\propto\pi_{ref}(y|x)\exp(r(x,y)/\beta)$. Odwracając to równanie, nagrodę można wyrazić przez politykę:

$$r(x,y)=\beta\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)$$

Podstawiając do modelu preferencji Bradleya-Terry'ego człon $Z(x)$ się skraca, co daje:

$$\mathcal L_{DPO}=-\mathbb E_{(x,y_w,y_l)}\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

Zwiększa on względne prawdopodobieństwo odpowiedzi preferowanej $y_w$ nad odrzuconą $y_l$ (względem modelu referencyjnego). $\beta$ (zwykle ok. 0,1-0,5) kontroluje siłę "kotwicy" do $\pi_{ref}$.

#### Procedura

1. SFT modelu → $\pi_{ref}$ (zamrożony).
2. Zbiór par (prompt, chosen, rejected).
3. Trening polityki z powyższą stratą (potrzebne log-prawdopodobieństwa z obu modeli).

#### Zalety względem RLHF/PPO

- Prosty, stabilny, jak zwykły fine-tuning nadzorowany; brak samplingu w pętli, brak reward modelu i krytyka.
- Mniejsze zużycie pamięci i mniej hiperparametrów.
- Dobra jakość w praktyce – szeroko stosowany w otwartych modelach.

#### Wady i pułapki

- Uczy się z **danych off-policy** (statycznych par), więc bywa słabszy od on-policy RL w trudnych zadaniach; możliwy spadek prawdopodobieństwa obu odpowiedzi.
- Skłonność do rozwlekłości i overfittingu preferencji; wrażliwość na jakość danych i $\beta$.
- Bez jawnego reward modelu trudniej diagnozować reward hacking.
- Warianty: IPO, KTO, ORPO, SimPO, online/iterative DPO.

```python
import torch.nn.functional as F
def dpo_loss(pi_w, pi_l, ref_w, ref_l, beta=0.1):
    logits = beta * ((pi_w - ref_w) - (pi_l - ref_l))
    return -F.logsigmoid(logits).mean()
```

Wejściem są sumy log-prawdopodobieństw tokenów odpowiedzi.

**Źródła:**
- [Direct Preference Optimization (DPO) (Outcome School)](https://outcomeschool.com/blog/direct-preference-optimization-dpo)
- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- [Hugging Face TRL – DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)
- [A General Theoretical Paradigm to Understand Learning from Human Preferences (IPO)](https://arxiv.org/abs/2310.12036)

---

<a id="q202"></a>
### 202. Group Relative Policy Optimization (GRPO).

**Odpowiedź:**

GRPO (DeepSeekMath, Shao et al., 2024) to wariant PPO **bez modelu krytyka (value model)**: linią bazową dla advantage jest **średnia nagroda w grupie odpowiedzi** wygenerowanych dla tego samego promptu. Ogranicza pamięć i złożoność, dlatego stał się podstawą treningu rozumowania (DeepSeek-R1).

#### Algorytm

Dla promptu $q$ próbkujemy grupę $G$ odpowiedzi $\{o_1,\dots,o_G\}\sim\pi_{\theta_{old}}$ i liczymy nagrody $r_1,\dots,r_G$ (np. weryfikowalna poprawność wyniku plus format). Advantage to znormalizowana nagroda względem grupy:

$$\hat A_i=\frac{r_i-\text{mean}(r_1..r_G)}{\text{std}(r_1..r_G)}$$

Cel (uproszczenie, z wersją tokenową):

$$J=\mathbb E\left[\frac1G\sum_{i=1}^G\frac1{|o_i|}\sum_t\Big(\min\big(\rho_{i,t}\hat A_i,\;\text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)-\beta\,\mathrm{KL}(\pi_\theta\|\pi_{ref})\Big)\right]$$

gdzie $\rho_{i,t}=\pi_\theta(o_{i,t}|q,o_{i,<t})/\pi_{\theta_{old}}(o_{i,t}|q,o_{i,<t})$. Kara KL trafia bezpośrednio do funkcji celu (a nie do nagrody jak w PPO-RLHF).

**Przykład**: $G=4$, nagrody $[1,0,0,1]$: średnia 0,5, odchylenie 0,5 → advantage $[+1,-1,-1,+1]$. Poprawne odpowiedzi są wzmacniane, niepoprawne osłabiane. Gdy wszystkie odpowiedzi mają tę samą nagrodę, advantage wynosi 0 (brak sygnału).

#### Zalety

- **Brak krytyka** – mniej pamięci (kluczowe przy dużych LLM) i prostsza implementacja.
- Naturalnie pasuje do **nagród weryfikowalnych** (matematyka, kod), gdzie porównanie w grupie jest wiarygodnym sygnałem.
- Stabilna normalizacja advantage.

#### Wady i uwagi

- Wymaga wielu próbek na prompt ($G$ często 8-64), więc generacja jest kosztowna.
- Normalizacja przez długość i odchylenie standardowe wprowadza obciążenia (np. preferencja długości) – późniejsze warianty (Dr. GRPO, DAPO) je korygują.
- Słabszy sygnał, gdy nagroda jest rzadka (grupy bez poprawnych odpowiedzi).

**Źródła:**
- [Group Relative Policy Optimization (GRPO) (Outcome School)](https://outcomeschool.com/blog/group-relative-policy-optimization-grpo)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability via RL](https://arxiv.org/abs/2501.12948)
- [Hugging Face TRL – GRPO Trainer](https://huggingface.co/docs/trl/grpo_trainer)

---

<a id="q203"></a>
### 203. Rekursywne modele językowe (Recursive Language Models, RLM).

**Odpowiedź:**

Termin "Recursive Language Models" w kontekście LLM oznacza **strategię inferencji**, w której model językowy nie przetwarza całego bardzo długiego wejścia w jednym oknie kontekstowym, lecz traktuje je jako **zewnętrzne środowisko** i sam **rekurencyjnie wywołuje siebie** (lub mniejszego pomocnika) na jego fragmentach. Opis poniżej opiera się na ogólnej idei i materiale Outcome School; szczegóły konkretnych implementacji mogą się różnić.

#### Problem

Nawet modele z ogromnym oknem cierpią na "context rot": jakość spada wraz z długością kontekstu (efekt "lost in the middle"), koszt rośnie, a okno i tak jest skończone. Zadania wymagające gęstego użycia informacji z całego wejścia (agregacja, porównania między dokumentami) są szczególnie trudne.

#### Idea

1. Długi prompt (dokumenty, logi, repozytorium) jest ładowany nie do kontekstu, ale do **środowiska programistycznego** (np. REPL Pythona) jako zmienna.
2. Model widzi tylko metadane (rozmiar, początek) i **pisze kod**, żeby ją eksplorować: szukać wyrażeniami regularnymi, dzielić na części, filtrować.
3. Dla wybranych fragmentów model **wywołuje rekurencyjnie sub-LLM** (podzadanie w świeżym, krótkim kontekście) i zbiera wyniki w zmiennych.
4. Na końcu składa odpowiedź z wyników cząstkowych.

Efektywny kontekst jest więc praktycznie nieograniczony, a każde pojedyncze wywołanie działa na krótkim, "czystym" kontekście.

#### Porównanie z podejściami pokrewnymi

| Podejście | Różnica |
|---|---|
| RAG | retrieval jest predefiniowany (embedding + top-k); w RLM model sam planuje eksplorację |
| Streszczanie/kompaktowanie | stratne; RLM ma dostęp do surowych danych |
| Agenci z narzędziami | RLM to szczególny przypadek z rekurencją i danymi poza oknem |
| Rozszerzanie okna (RoPE scaling) | nie rozwiązuje zaniku jakości i kosztu |

#### Kompromisy

- Wiele wywołań → wyższa łączna latencja i koszt na łatwych zadaniach, nieprzewidywalna liczba kroków.
- Zależność od zdolności modelu do pisania poprawnego kodu i planowania; ryzyko błędów agregacji.
- Wymaga bezpiecznego sandboxa do wykonywania kodu.

**Źródła:**
- [Recursive Language Models (Outcome School)](https://outcomeschool.com/blog/recursive-language-models)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)

---

<a id="q204"></a>
### 204. Uczenie ciągłe (Continual Learning) w LLM.

**Odpowiedź:**

Continual learning to zdolność modelu do **uczenia się nowych danych, zadań i wiedzy w czasie bez utraty tego, co już umie**. Głównym wyzwaniem jest **catastrophic forgetting**: aktualizacja wag pod nowe dane nadpisuje reprezentacje istotne dla starych zadań. Modele LLM po pretrainingu mają statyczną wiedzę z określonym "knowledge cutoff", a ich ponowne trenowanie od zera jest bardzo drogie.

#### Scenariusze w LLM

- **Continual pretraining / domain-adaptive pretraining**: dalszy pretraining na nowych danych (np. prawo, medycyna, nowy język, świeże wiadomości).
- **Continual instruction tuning / alignment**: kolejne zadania i preferencje.
- **Aktualizacja wiedzy** (knowledge editing): korekta pojedynczych faktów.

#### Techniki

| Rodzina | Idea | Przykłady |
|---|---|---|
| Replay | mieszanie nowych danych z próbką starych (lub syntetycznych) | experience replay, generative replay |
| Regularyzacja | kara za zmianę ważnych wag | EWC (Fisher information), L2-SP, KL do starego modelu |
| Izolacja parametrów / architektura | osobne parametry dla nowych zadań, stare zamrożone | LoRA/adaptery per zadanie, prompt/prefix tuning, MoE, model merging |
| Kontrola treningu | mały learning rate, re-warmup i ponowne schładzanie LR | strategie continual pretraining |
| Pamięć zewnętrzna | wiedza poza wagami | **RAG**, pamięć agenta, bazy wiedzy |

Model merging (uśrednianie wag, task arithmetic) pozwala łączyć wyspecjalizowane modele bez ponownego treningu.

#### Praktyka

- Dla **świeżej wiedzy faktograficznej** zwykle prostsze i bezpieczniejsze niż aktualizacja wag jest **RAG**.
- Przy fine-tuningu domenowym: mieszaj dane ogólne (replay), używaj LoRA i niskiego LR, monitoruj regresje na benchmarkach ogólnych i wcześniejszych zadaniach.
- Ewaluacja: mierz *forgetting* (spadek na starych zadaniach), *forward/backward transfer*, a nie tylko wynik na nowym zbiorze.

#### Otwarte problemy

Kompromis stabilność–plastyczność, uczenie online bez jawnych granic zadań, skalowalność, ocena wpływu na bezpieczeństwo (zgubienie alignmentu po dalszym fine-tuningu).

**Źródła:**
- [Continual Learning in LLMs (Outcome School)](https://outcomeschool.com/blog/continual-learning-in-llms)
- [Overcoming catastrophic forgetting in neural networks (EWC, Kirkpatrick et al.)](https://arxiv.org/abs/1612.00796)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Simple and Scalable Strategies to Continually Pre-train LLMs](https://arxiv.org/abs/2403.08763)
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

---

<a id="q205"></a>
### 205. Jak działa destylacja wiedzy (Knowledge Distillation)?

**Odpowiedź:**

Destylacja wiedzy to technika kompresji modeli, w której duży, dokładny model (**teacher**) uczy mniejszy model (**student**) naśladowania swoich wyjść. Student uczy się nie tylko z twardych etykiet, ale też z **miękkich rozkładów prawdopodobieństwa** nauczyciela, które zawierają dodatkową informację (tzw. *dark knowledge*): np. że obraz kota jest „bardziej podobny" do psa niż do samochodu.

#### Mechanizm

Logity nauczyciela $z^T$ i ucznia $z^S$ są zmiękczane parametrem temperatury $T$:

$$p_i^{(T)} = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

Funkcja straty to zwykle kombinacja:

$$\mathcal{L} = \alpha \, T^2 \, \mathrm{KL}\big(p^{T}_{teacher} \,\|\, p^{T}_{student}\big) + (1-\alpha)\, \mathrm{CE}(y, p_{student})$$

Czynnik $T^2$ kompensuje zmniejszenie gradientów przy wysokiej temperaturze. Wyższe $T$ daje bardziej „płaski" rozkład i ujawnia relacje między klasami.

#### Warianty

- **Response-based** (logit distillation): dopasowanie wyjść, wariant klasyczny (Hinton et al.).
- **Feature-based**: dopasowanie ukrytych reprezentacji pośrednich (np. FitNets), często z warstwą projekcji.
- **Relation-based**: dopasowanie relacji między próbkami lub warstwami.
- **Sequence-level / data distillation w LLM**: nauczyciel (np. duży LLM) generuje odpowiedzi lub rozumowania (chain-of-thought), a student jest fine-tunowany na tych syntetycznych danych (supervised fine-tuning). Tak działa wiele małych modeli instrukcyjnych.
- **Self-distillation**: nauczyciel i uczeń mają tę samą architekturę.
- **Online distillation**: oba modele trenowane równocześnie.

#### Przykład: DistilBERT

DistilBERT redukuje BERT-a o ok. 40% parametrów, zachowując zdecydowaną większość jakości i przyspieszając inferencję (wg autorów ok. 60%), dzięki stracie łączącej masked LM, destylację i cosinusowe dopasowanie stanów ukrytych.

#### Kod (PyTorch)

```python
import torch.nn.functional as F

def kd_loss(s_logits, t_logits, y, T=2.0, alpha=0.5):
    soft = F.kl_div(F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1),
                    reduction="batchmean") * (T * T)
    hard = F.cross_entropy(s_logits, y)
    return alpha * soft + (1 - alpha) * hard
```

#### Zalety, wady, pułapki

- Mniejszy koszt inferencji i pamięci, łatwiejszy deployment na urządzeniach brzegowych.
- Student rzadko przewyższa nauczyciela; luka pojemności (za duży nauczyciel względem ucznia) może pogarszać wyniki, pomaga *teacher assistant*.
- Trzeba dobrać $T$ i $\alpha$; nauczyciel musi być dobrze skalibrowany.
- W przypadku modeli komercyjnych warto sprawdzić licencję (zakaz używania wyjść do trenowania konkurencyjnych modeli).
- Destylację można łączyć z kwantyzacją i pruningiem.

**Źródła:**
- [How does Knowledge Distillation work? (Outcome School)](https://outcomeschool.com/blog/how-does-knowledge-distillation-work)
- [Distilling the Knowledge in a Neural Network (Hinton et al.)](https://arxiv.org/abs/1503.02531)
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)

---

<a id="q206"></a>
### 206. Czym jest instruction tuning i dlaczego jest ważny dla modeli czatowych?

**Odpowiedź:**

**Instruction tuning** (nadzorowany fine-tuning na instrukcjach, SFT) to dotrenowanie wstępnie wytrenowanego modelu językowego na zbiorze par *(instrukcja, pożądana odpowiedź)*. Model bazowy (pretrained) został wytrenowany tylko do przewidywania następnego tokenu, więc na prompt „Napisz e-mail z przeprosinami" może po prostu kontynuować tekst w stylu internetowego forum, zamiast wykonać polecenie. Instruction tuning uczy go **formatu interakcji**: rozumienia poleceń, odpowiadania na nie i zachowania zgodnego z intencją użytkownika.

#### Jak to wygląda

- **Dane**: ludzkie demonstracje (annotatorzy piszą wzorcowe odpowiedzi), zbiory zadań NLP przekształcone w instrukcje (FLAN), dane syntetyczne generowane przez silniejsze modele (Self-Instruct, Alpaca).
- **Format**: szablon czatu z rolami (system / user / assistant) i specjalnymi tokenami.
- **Strata**: standardowa cross-entropy na tokenach odpowiedzi; tokeny promptu zwykle są **maskowane** (loss tylko na odpowiedzi).
- **Skala**: często wystarcza od tysięcy do setek tysięcy przykładów; wg pracy LIMA jakość danych liczy się bardziej niż ilość.
- **PEFT**: często realizowany przez LoRA/QLoRA zamiast pełnego fine-tuningu.

#### Miejsce w potoku (InstructGPT)

1. Pretraining (ogromny korpus).
2. **SFT / instruction tuning** na demonstracjach.
3. Trening modelu nagrody na preferencjach ludzi.
4. Optymalizacja polityki (RLHF/PPO, dziś też DPO, GRPO).

Kluczowy wynik InstructGPT: mały model po SFT+RLHF (1,3B) był przez ludzi oceniany wyżej niż GPT-3 175B, choć był ok. 100 razy mniejszy.

#### Dlaczego to ważne

- Zamienia „autouzupełnianie" w asystenta wykonującego polecenia (zero-shot generalizacja na nowe zadania, potwierdzona w FLAN).
- Poprawia zgodność z intencją (alignment), format odpowiedzi, kontrolę tonu.
- Tworzy bazę pod dalszy alignment (RLHF/DPO).

#### Pułapki

- Model uczy się stylu, a niekoniecznie nowej wiedzy; SFT na faktach, których model nie zna, może zwiększać halucynacje.
- Niska jakość lub niska różnorodność danych, przeuczenie, catastrophic forgetting.
- Zatrucie danych i uprzedzenia w annotacjach.

**Źródła:**
- [Decoding InstructGPT (Outcome School)](https://outcomeschool.com/blog/decoding-instructgpt)
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)

---

<a id="q207"></a>
### 207. Prefill vs Decode: czym się różnią fazy inferencji LLM?

**Odpowiedź:**

Generowanie tekstu przez LLM (autoregresyjne, dekoderowe) składa się z dwóch faz o zupełnie różnej charakterystyce obliczeniowej.

#### Prefill (przetwarzanie promptu)

- Cały prompt ($n$ tokenów) jest przetwarzany **równolegle** w jednym przejściu do przodu.
- Buduje **KV cache**: klucze i wartości uwagi dla każdej warstwy.
- Duże mnożenia macierzy, wysoka intensywność arytmetyczna: faza **compute-bound**.
- Wyznacza pierwszy token wyjściowy; jej czas to główny składnik **TTFT** (Time To First Token).
- Koszt uwagi rośnie kwadratowo z długością promptu.

#### Decode (generowanie)

- Tokeny powstają **jeden po drugim**; każdy krok przetwarza tylko jeden nowy token, ale musi odczytać wszystkie wagi modelu oraz cały KV cache.
- Mała intensywność arytmetyczna: faza **memory-bandwidth-bound**, GPU jest słabo wykorzystane przy małym batchu.
- Wyznacza **TPOT / ITL** (czas na token wyjściowy), a więc płynność streamingu.
- KV cache rośnie liniowo z długością sekwencji, co zjada pamięć GPU.

| Cecha | Prefill | Decode |
|---|---|---|
| Równoległość | po tokenach promptu | brak (sekwencyjnie) |
| Ograniczenie | obliczenia (FLOPs) | przepustowość pamięci |
| Metryka | TTFT | TPOT, throughput |
| Rozmiar batcha | mało żądań wystarczy do nasycenia | potrzeba dużego batcha |

#### Optymalizacje

- **Continuous batching** i **PagedAttention** (vLLM): lepsze użycie pamięci i throughput.
- **Chunked prefill** (np. Sarathi-Serve): dzielenie długich promptów na kawałki mieszane z krokami decode, aby prefill nie blokował generowania.
- **Disaggregated serving** (DistServe): osobne pule GPU dla prefill i decode, bo mają inne wymagania.
- **Prefix caching**: ponowne użycie KV cache wspólnego prefiksu (np. system prompt).
- **Speculative decoding**, kwantyzacja wag i KV cache, GQA/MQA (mniejszy KV cache), FlashAttention.

#### Praktyczna wskazówka

Długi kontekst wejściowy i krótka odpowiedź (RAG, streszczenia) obciążają prefill; krótki prompt i długa odpowiedź (generowanie kodu) obciążają decode. SLO należy formułować osobno dla TTFT i TPOT.

**Źródła:**
- [Prefill vs Decode: LLM Inference Optimization (Outcome School)](https://outcomeschool.com/blog/prefill-vs-decode-llm-inference-optimization)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference](https://arxiv.org/abs/2403.02310)
- [DistServe: Disaggregating Prefill and Decoding](https://arxiv.org/abs/2401.09670)

---

<a id="q208"></a>
### 208. Jak działa Sliding Window Attention?

**Odpowiedź:**

Standardowa uwaga (full self-attention) ma złożoność $O(n^2)$ względem długości sekwencji $n$ w czasie i pamięci. **Sliding Window Attention (SWA)** ogranicza uwagę tak, by każdy token widział tylko **$W$ ostatnich tokenów** (okno lokalne) zamiast całego kontekstu.

#### Mechanizm

Dla pozycji $i$ maska pozwala na uwagę do pozycji $j$ takich, że $i-W < j \le i$ (w wersji przyczynowej):

$$\text{Attn}(i) = \mathrm{softmax}\!\left(\frac{q_i K_{i-W+1:i}^\top}{\sqrt{d}}\right) V_{i-W+1:i}$$

- Złożoność spada do $O(n \cdot W)$, czyli liniowo względem $n$.
- **Efektywne pole recepcyjne rośnie z głębokością**: token w warstwie $l$ pośrednio zawiera informację z ok. $l \cdot W$ pozycji wstecz, bo warstwa $l$ czyta stany warstwy $l-1$, które już zagregowały swoje okna (analogia do splotów w CNN).
- Podczas dekodowania KV cache może być **buforem cyklicznym (rolling buffer)** o stałym rozmiarze $W$, więc pamięć nie rośnie z długością generacji.

#### Zastosowania

- **Longformer** łączy okno lokalne z kilkoma tokenami globalnymi.
- **Mistral 7B** używa SWA (okno 4096) razem z GQA; pozwala to obsługiwać dłuższe sekwencje przy mniejszym koszcie.
- Wiele nowszych modeli miesza warstwy z pełną uwagą i warstwy z oknem (hybrydy lokalne/globalne).

#### Zalety i wady

- (+) Liniowy koszt obliczeń, stała pamięć KV cache, szybsza inferencja na długich tekstach.
- (-) Informacja spoza okna jest dostępna tylko pośrednio i ulega degradacji (nie ma gwarancji dokładnego wyszukiwania odległych faktów, np. *needle in a haystack*).
- (-) Wymaga wydajnych kerneli (FlashAttention obsługuje okna), inaczej zysk teoretyczny nie przekłada się na praktykę.
- Modele trenowane z pełną uwagą i przełączone na okno bez adaptacji zwykle działają źle, co prowadzi do zjawiska attention sinks (patrz następne pytanie).

**Źródła:**
- [How does Sliding Window Attention work? (Outcome School)](https://outcomeschool.com/blog/how-does-sliding-window-attention-work)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- [Mistral 7B](https://arxiv.org/abs/2310.06825)

---

<a id="q209"></a>
### 209. Jak działają Attention Sinks?

**Odpowiedź:**

**Attention sink** to zjawisko, w którym modele autoregresyjne przypisują **nieproporcjonalnie dużą wagę uwagi kilku pierwszym tokenom** sekwencji (często już samemu pierwszemu), nawet jeśli nie są one semantycznie istotne. Ponieważ softmax wymusza sumowanie wag do 1, głowa uwagi, która „nie ma nic użytecznego do przeczytania", musi gdzieś zrzucić masę prawdopodobieństwa. Pierwsze tokeny są widoczne dla wszystkich kolejnych pozycji w maskowanej uwadze przyczynowej, więc naturalnie stają się takim „zlewem".

#### Problem, który to ujawnia

Przy strumieniowym generowaniu z oknem przesuwnym (usuwanie najstarszych wpisów z KV cache) jakość modelu **gwałtownie się załamuje** (perplexity rośnie), gdy tylko pierwsze tokeny wypadną z okna. Usunięcie sinków zaburza rozkład softmax, a nie tylko traci kontekst.

#### StreamingLLM

Xiao et al. proponują prosty schemat bez dotrenowywania:

- zachowaj w KV cache **kilka początkowych tokenów (np. 4)** jako attention sinks,
- plus **okno ostatnich tokenów** (rolling window),
- pozycje przypisuj względem pozycji w cache, a nie oryginalnych indeksów tekstu.

Wynik: stabilne generowanie (w pracy dla milionów tokenów) o stałej pamięci i opóźnieniu. Uwaga: to **nie rozszerza efektywnego kontekstu**, model nie pamięta tego, co wypadło z okna.

#### Dodatkowe pomysły

- Trening z dedykowanym **tokenem sink** (placeholder), aby model miał jawny zlew.
- Modyfikacje softmax (np. „softmax+1", doliczenie stałego składnika do mianownika), aby głowa mogła „nic nie wybrać".
- Wariant wykorzystywany w niektórych nowszych modelach (np. uczone parametry sink per-head).

#### Praktyczne wnioski

- Przy własnym implementowaniu eviction w KV cache nigdy nie usuwaj pierwszych tokenów.
- Sinks tłumaczą, czemu SWA działa w praktyce, gdy w cache zostaje początek sekwencji.
- Zjawisko utrudnia też kwantyzację (masywne aktywacje), co warto uwzględnić.

**Źródła:**
- [How do Attention Sinks work? (Outcome School)](https://outcomeschool.com/blog/how-do-attention-sinks-work)
- [Efficient Streaming Language Models with Attention Sinks (StreamingLLM)](https://arxiv.org/abs/2309.17453)
- [Mistral 7B (Sliding Window Attention)](https://arxiv.org/abs/2310.06825)

---

## Ewaluacja modeli

<a id="q210"></a>
### 210. Czym są precision, recall, F1 score i accuracy?

**Odpowiedź:**

Metryki te opisują jakość klasyfikatora binarnego na podstawie macierzy pomyłek: TP (prawdziwie pozytywne), FP (fałszywie pozytywne), TN (prawdziwie negatywne), FN (fałszywie negatywne).

| Metryka | Wzór | Pytanie, na które odpowiada |
|---|---|---|
| Accuracy | $\frac{TP+TN}{TP+TN+FP+FN}$ | Jaki odsetek wszystkich predykcji jest poprawny? |
| Precision | $\frac{TP}{TP+FP}$ | Ile z przewidzianych pozytywów jest naprawdę pozytywnych? |
| Recall (sensitivity, TPR) | $\frac{TP}{TP+FN}$ | Ile z prawdziwych pozytywów wykryliśmy? |
| F1 | $\frac{2PR}{P+R}$ | Średnia harmoniczna precision i recall |

#### Intuicja i kompromis

- **Precision** ważne, gdy fałszywy alarm jest kosztowny (filtr spamu usuwający ważne maile, rekomendacja leku).
- **Recall** ważne, gdy przeoczenie jest kosztowne (diagnostyka nowotworów, wykrywanie oszustw).
- Podnosząc próg decyzyjny, zwykle zyskujemy precision kosztem recall i odwrotnie.
- **F1** to średnia harmoniczna, która karze duże różnice między P i R (średnia arytmetyczna 0,9 i 0,1 daje 0,5, ale F1 tylko 0,18). Uogólnienie: $F_\beta = (1+\beta^2)\frac{PR}{\beta^2 P + R}$, gdzie $\beta>1$ faworyzuje recall.

#### Przykład liczbowy

Zbiór 1000 próbek, 10 pozytywnych. Model przewiduje wszystko jako negatywne: accuracy = 99%, ale recall = 0. To klasyczna pułapka accuracy przy niezbalansowanych klasach. Inny model: TP=8, FP=12, FN=2, TN=978 daje precision = 0,4, recall = 0,8, F1 ≈ 0,533, accuracy = 98,6%.

#### Kod

```python
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
acc = accuracy_score(y_true, y_pred)
```

#### Pułapki

- F1 ignoruje TN, więc nie nadaje się, gdy klasa negatywna też ma znaczenie.
- Przy wielu klasach trzeba wybrać uśrednianie: macro, micro, weighted.
- Zawsze raportuj metryki razem z rozkładem klas i progiem.

**Źródła:**
- [Precision and recall (Wikipedia)](https://en.wikipedia.org/wiki/Precision_and_recall)
- [scikit-learn: Model evaluation, classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [scikit-learn: Precision, recall and F-measures](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)

---

<a id="q211"></a>
### 211. Czym jest macierz pomyłek (confusion matrix) i jak ją interpretować?

**Odpowiedź:**

Macierz pomyłek to tabela zestawiająca **rzeczywiste klasy** z **klasami przewidzianymi** przez model. Dla klasyfikacji binarnej:

| | Przewidziane: 1 | Przewidziane: 0 |
|---|---|---|
| **Rzeczywiste: 1** | TP | FN |
| **Rzeczywiste: 0** | FP | TN |

(Uwaga: scikit-learn zwraca macierz w kolejności `[[TN, FP], [FN, TP]]`, wiersze to klasy rzeczywiste, kolumny to przewidziane, klasy posortowane rosnąco.)

#### Interpretacja

- **Przekątna** to poprawne predykcje; wszystko poza nią to błędy.
- **FP** (błąd I rodzaju) i **FN** (błąd II rodzaju) mają zwykle różny koszt biznesowy.
- Z macierzy wyprowadzamy: accuracy, precision, recall, specificity ($TN/(TN+FP)$), FPR, F1, MCC.
- Dla wielu klas macierz ma rozmiar $K\times K$; komórka $(i,j)$ mówi, ile próbek klasy $i$ zaklasyfikowano jako $j$. Pozwala zobaczyć, które klasy model **ze sobą myli** (np. kot vs rys), co jest bezcenne w analizie błędów.
- **Normalizacja** (po wierszach, `normalize="true"`) pokazuje recall per klasa i ułatwia analizę przy niezbalansowanych danych.

#### Kod

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred, normalize="true")
ConfusionMatrixDisplay(cm).plot()
```

#### Praktyczne wskazówki

- Analizuj macierz przy różnych progach decyzyjnych; wybór progu to wybór punktu równowagi FP/FN.
- W problemach wieloetykietowych używa się macierzy per etykieta (`multilabel_confusion_matrix`).
- Sama macierz nie mówi, *dlaczego* model się myli; uzupełnij ją przeglądem konkretnych błędnych przykładów.

**Źródła:**
- [Confusion matrix (Wikipedia)](https://en.wikipedia.org/wiki/Confusion_matrix)
- [scikit-learn: confusion_matrix](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Confusion Matrix (post Pallavi Shekhar, LinkedIn)](https://www.linkedin.com/posts/pallavi-shekhar_ai-ml-activity-7438197172992606208-oEUc)

---

<a id="q212"></a>
### 212. Jakie są typowe metryki ewaluacji w klasyfikacji?

**Odpowiedź:**

Wybór metryki zależy od problemu (kosztów błędów, balansu klas, tego, czy potrzebne są prawdopodobieństwa).

#### Metryki oparte na twardych etykietach

- **Accuracy**: odsetek poprawnych predykcji; mylące przy niezbalansowanych klasach.
- **Precision, Recall, F1 / $F_\beta$**.
- **Specificity** $=TN/(TN+FP)$.
- **Balanced accuracy**: średnia recall po klasach.
- **MCC** (Matthews Correlation Coefficient): uwzględnia wszystkie 4 pola macierzy, wiarygodny przy niezbalansowaniu.
- **Cohen's kappa**: zgodność skorygowana o przypadkową.

#### Metryki oparte na prawdopodobieństwach / rankingu

- **ROC AUC**: zdolność rankingowania pozytywów nad negatywami, niezależna od progu.
- **PR AUC / Average Precision**: lepsza przy silnym niezbalansowaniu.
- **Log loss (cross-entropy)**: ocenia jakość prawdopodobieństw:
$$-\frac1N\sum_i \big[y_i\log p_i + (1-y_i)\log(1-p_i)\big]$$
- **Brier score**: błąd kwadratowy prawdopodobieństw.
- **Calibration** (krzywe kalibracji, ECE): czy 0,8 znaczy naprawdę ~80%.

#### Metryki dla wielu klas / etykiet

- Macro / micro / weighted F1, top-k accuracy, Hamming loss, Jaccard, subset accuracy.

#### Metryki decyzyjne

- **Koszt oczekiwany** (macierz kosztów), **lift**, **gain**, precision@k, recall@k przy ograniczonej pojemności (np. zespół może sprawdzić 100 przypadków dziennie).

#### Zasady doboru

1. Zdefiniuj koszt FP vs FN.
2. Sprawdź balans klas.
3. Zdecyduj, czy potrzebne są prawdopodobieństwa (kalibracja) czy samo rankowanie.
4. Raportuj kilka metryk i przedziały ufności.

**Źródła:**
- [scikit-learn: Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Evaluation of binary classifiers (Wikipedia)](https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers)
- [Matthews correlation coefficient (Wikipedia)](https://en.wikipedia.org/wiki/Phi_coefficient)

---

<a id="q213"></a>
### 213. Kiedy używać accuracy, a kiedy innych metryk?

**Odpowiedź:**

**Accuracy** ma sens, gdy: klasy są w przybliżeniu zrównoważone, koszty FP i FN są podobne, a interesuje nas ogólny odsetek poprawnych decyzji (np. rozpoznawanie cyfr MNIST, klasyfikacja tematów o równych klasach).

#### Kiedy accuracy zawodzi

- **Niezbalansowane klasy**: przy 1% oszustw model „zawsze OK" ma 99% accuracy i jest bezużyteczny.
- **Asymetryczne koszty**: pominięcie nowotworu jest gorsze niż fałszywy alarm.
- Gdy ważna jest **jakość prawdopodobieństw** lub ranking, a nie twarda etykieta.

#### Co wybrać zamiast

| Sytuacja | Metryka |
|---|---|
| Kosztowny fałszywy alarm | Precision (specificity) |
| Kosztowne przeoczenie | Recall, $F_2$ |
| Silny imbalance, ocena pozytywów | PR AUC, F1, MCC |
| Ranking niezależny od progu | ROC AUC |
| Wiarygodne prawdopodobieństwa | Log loss, Brier, kalibracja |
| Zbalansowanie recall po klasach | Balanced accuracy, macro F1 |
| Ograniczona pojemność przeglądu | precision@k, lift |

#### Przykład

Wykrywanie oszustw: 10 000 transakcji, 50 oszustw. Model A wykrywa 40 oszustw z 100 alarmami (precision 0,4, recall 0,8), model B nie wykrywa niczego. Accuracy: A = 99,4%, B = 99,5%, więc accuracy wskazuje na złego zwycięzcę.

#### Zalecenia

- Zawsze porównuj z **baseline** (klasa większościowa) i raportuj accuracy tylko razem z innymi metrykami.
- Wiąż metrykę techniczną z metryką biznesową (koszt, przychód, SLA).

**Źródła:**
- [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Accuracy paradox (Wikipedia)](https://en.wikipedia.org/wiki/Accuracy_paradox)
- [scikit-learn: Balanced accuracy score](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score)

---

<a id="q214"></a>
### 214. Kiedy używać log loss zamiast accuracy?

**Odpowiedź:**

**Log loss** (binary/categorical cross-entropy) mierzy jakość **prawdopodobieństw** przewidywanych przez model, a nie tylko trafność twardej etykiety:

$$\text{LogLoss} = -\frac1N\sum_{i=1}^N\sum_{k=1}^K y_{ik}\log p_{ik}$$

Kara jest bardzo duża za **pewne, ale błędne** predykcje: przy prawdziwej klasie 1 predykcja $p=0{,}01$ kosztuje $-\ln 0{,}01\approx 4{,}6$, a $p=0{,}4$ tylko $\approx 0{,}92$. Accuracy tego nie rozróżnia: obie są „błędne" przy progu 0,5.

#### Kiedy log loss

- Gdy **prawdopodobieństwa są używane dalej**: scoring ryzyka kredytowego, CTR w reklamie, decyzje oparte na wartości oczekiwanej, ubezpieczenia.
- Gdy istotna jest **kalibracja** (model mówiący 0,7 powinien mieć rację w ~70% przypadków). Log loss jest *strictly proper scoring rule*, więc minimalizowany jest przez prawdziwe prawdopodobieństwa.
- Do **porównywania modeli** z bardziej gładkim sygnałem niż accuracy (accuracy zmienia się skokowo).
- Jako **funkcja straty i metryka early stopping** (spójna z treningiem).
- Konkursy (Kaggle) wymagające prawdopodobieństw.

#### Kiedy accuracy

- Gdy liczy się tylko trafność decyzji przy stałym progu, klasy zbalansowane, koszty symetryczne, i łatwo to komunikować interesariuszom.

#### Pułapki log loss

- Wrażliwy na skrajne błędy i **etykiety błędne (label noise)**; stosuje się clipping prawdopodobieństw ($\epsilon$).
- Trudniej interpretować niż accuracy; warto podać punkt odniesienia: model stały (prior) daje log loss równy entropii klas.
- Model o niższym log loss może mieć niższą accuracy i odwrotnie.
- Przy niezbalansowaniu warto sprawdzić dodatkowo kalibrację (`CalibratedClassifierCV`, isotonic/Platt).

**Źródła:**
- [scikit-learn: Log loss](https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss)
- [Scoring rule (Wikipedia)](https://en.wikipedia.org/wiki/Scoring_rule)
- [On Calibration of Modern Neural Networks (Guo et al.)](https://arxiv.org/abs/1706.04599)

---

<a id="q215"></a>
### 215. Jakich metryk użyjesz w klasyfikacji wieloklasowej?

**Odpowiedź:**

W klasyfikacji wieloklasowej ($K>2$) metryki binarne uogólnia się przez **uśrednianie** wyników per klasa (podejście one-vs-rest).

#### Uśrednianie

- **Macro**: średnia arytmetyczna metryki po klasach; każda klasa liczy się tak samo, więc wrażliwa na jakość klas rzadkich.
- **Micro**: agregacja TP/FP/FN po wszystkich klasach, potem obliczenie metryki. W klasyfikacji jednoetykietowej micro-precision = micro-recall = micro-F1 = accuracy.
- **Weighted**: średnia ważona liczebnością klas (odzwierciedla rozkład, ale maskuje słabe klasy rzadkie).

#### Główne metryki

- **Accuracy** i **balanced accuracy**.
- **Macro / weighted F1**, per-class precision i recall (`classification_report`).
- **Top-k accuracy** (np. top-5 w ImageNet) tam, gdzie kilka odpowiedzi jest akceptowalnych.
- **Log loss** (categorical cross-entropy) dla prawdopodobieństw.
- **ROC AUC** wieloklasowy: `roc_auc_score(..., multi_class="ovr" lub "ovo")`.
- **Cohen's kappa**, **MCC** (uogólniony).
- **Macierz pomyłek** $K\times K$ do analizy, które klasy są mylone.

#### Kod

```python
from sklearn.metrics import classification_report, f1_score, roc_auc_score
print(classification_report(y_true, y_pred, digits=3))
f1_macro = f1_score(y_true, y_pred, average="macro")
auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
```

#### Wskazówki

- Gdy klasy są niezbalansowane i wszystkie ważne: macro F1. Gdy ważny ogólny odsetek: micro/accuracy.
- Przy hierarchii klas rozważ metryki hierarchiczne lub koszt zależny od odległości klas.
- Dla wielu etykiet (multilabel) używaj Hamming loss, samples-F1, Jaccard.

**Źródła:**
- [scikit-learn: Multiclass and multilabel classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel)
- [scikit-learn: classification_report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [scikit-learn: roc_auc_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)

---

<a id="q216"></a>
### 216. Jak radzić sobie z niezbalansowaniem klas w metrykach klasyfikacji?

**Odpowiedź:**

Przy niezbalansowanych danych (np. 1% klasy pozytywnej) accuracy jest zwodnicza. Trzeba dobrać metryki, które nie są zdominowane przez klasę większościową.

#### Odpowiednie metryki

- **Precision, recall, F1** dla klasy mniejszościowej (nie uśredniaj z klasą dominującą).
- **PR AUC / Average Precision**: bardziej informatywna niż ROC AUC, bo nie używa TN; przy dużej liczbie negatywów ROC AUC może wyglądać optymistycznie mimo niskiej precision.
- **Balanced accuracy**, **macro F1**, **MCC**, **Cohen's kappa**.
- **Precision@k / recall@k**, **lift**, gdy pojemność akcji jest ograniczona.
- **Koszt oczekiwany** z macierzą kosztów.

#### Dobre praktyki ewaluacji

- **Stratyfikowany podział** train/val/test i stratified k-fold, aby zachować proporcje klas.
- **Nie stosuj oversamplingu przed podziałem** (data leakage); resampling tylko na zbiorze treningowym, w obrębie folda.
- Zbiór testowy zostaje z **naturalnym rozkładem** klas, żeby metryki odzwierciedlały produkcję.
- Dobierz **próg decyzyjny** na zbiorze walidacyjnym (krzywa PR, maksymalizacja $F_\beta$ lub minimalizacja kosztu), a nie domyślne 0,5.
- Raportuj **przedziały ufności** (bootstrap), bo przy mało próbkach mniejszościowych wyniki są bardzo niestabilne.
- Sprawdź **kalibrację** po zastosowaniu wag lub resamplingu, bo zmienia się prior.

#### Kod

```python
from sklearn.metrics import average_precision_score, precision_recall_curve
ap = average_precision_score(y_true, proba)
prec, rec, thr = precision_recall_curve(y_true, proba)
f1 = 2 * prec * rec / (prec + rec + 1e-12)
best_thr = thr[f1[:-1].argmax()]
```

**Źródła:**
- [The Relationship Between Precision-Recall and ROC Curves (Davis & Goadrich)](https://www.biostat.wisc.edu/~page/rocpr.pdf)
- [scikit-learn: average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)

---

<a id="q217"></a>
### 217. Czym jest krzywa ROC? Czym jest AUC?

**Odpowiedź:**

**Krzywa ROC** (Receiver Operating Characteristic) pokazuje, jak zmieniają się **TPR** (recall) oraz **FPR** przy zmianie progu decyzyjnego klasyfikatora probabilistycznego:

$$TPR = \frac{TP}{TP+FN},\qquad FPR = \frac{FP}{FP+TN}$$

Każdy próg to jeden punkt $(FPR, TPR)$. Punkt $(0,0)$: wszystko negatywne; $(1,1)$: wszystko pozytywne; $(0,1)$: model idealny; przekątna: losowy klasyfikator.

#### AUC

**AUC** (Area Under the Curve) to pole pod krzywą ROC, wartość w $[0,1]$:

- 0,5 = losowy model, 1,0 = idealny, <0,5 = odwrócone predykcje.
- **Interpretacja probabilistyczna**: AUC to prawdopodobieństwo, że losowo wybrany przykład pozytywny dostanie wyższy score niż losowo wybrany negatywny (związek z testem Manna-Whitneya).
- Jest **niezależne od progu** oraz od skali scorów (liczy się tylko ranking).

#### Budowa krzywej

Sortujemy przykłady po score malejąco i przesuwamy próg, aktualizując TP/FP.

```python
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thr = roc_curve(y_true, scores)
auc = roc_auc_score(y_true, scores)
```

#### Zalety i ograniczenia

- (+) Porównanie modeli niezależnie od progu; niewrażliwe na zmianę prevalence przy stałych rozkładach scorów.
- (-) Przy silnym niezbalansowaniu może być zbyt optymistyczne, bo FPR jest rozcieńczony ogromną liczbą negatywów; wtedy użyj krzywej **PR**.
- (-) Nie mówi nic o kalibracji ani o optymalnym progu; dwa modele o tym samym AUC mogą się różnić w interesującym zakresie FPR (rozważ **partial AUC**).
- (-) Dla wielu klas: uśrednianie OvR/OvO.
- Optymalny próg wybiera się np. z indeksu Youdena ($TPR-FPR$) lub z kosztów.

**Źródła:**
- [Receiver operating characteristic (Wikipedia)](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [scikit-learn: Receiver Operating Characteristic (ROC)](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
- [An introduction to ROC analysis (Fawcett, 2006)](https://doi.org/10.1016/j.patrec.2005.10.010)

---

<a id="q218"></a>
### 218. Jak radzić sobie z niezbalansowanymi zbiorami danych?

**Odpowiedź:**

Niezbalansowanie (class imbalance) oznacza, że jedna klasa (mniejszościowa) występuje znacznie rzadziej niż inne (oszustwa, choroby, awarie). Model minimalizujący zwykły loss uczy się głównie klasy dominującej. Rozwiązania działają na trzech poziomach.

#### 1. Poziom danych

- **Random oversampling** klasy mniejszościowej (duplikaty): proste, ale ryzyko overfittingu.
- **Random undersampling** większościowej: szybkie, ale traci informację.
- **SMOTE** i warianty (Borderline-SMOTE, ADASYN): syntetyczne przykłady przez interpolację między sąsiadami klasy mniejszościowej. Uwaga na dane kategoryczne i szum.
- **Zebranie większej ilości danych** mniejszości, augmentacja (obrazy, tekst).
- Czyszczenie granicy klas (Tomek links, ENN).

#### 2. Poziom algorytmu

- **Wagi klas** (`class_weight="balanced"`, `scale_pos_weight` w XGBoost, `pos_weight` w `BCEWithLogitsLoss`).
- **Focal loss**: $-(1-p_t)^\gamma\log p_t$ zmniejsza wagę łatwych przykładów (RetinaNet).
- Metody zespołowe: BalancedRandomForest, EasyEnsemble, bagging z undersamplingiem.
- **Wykrywanie anomalii / one-class** przy ekstremalnej rzadkości.

#### 3. Poziom decyzji i ewaluacji

- Metryki: PR AUC, F1, MCC, recall@precision, koszt (nie accuracy).
- **Dostosowanie progu** decyzyjnego na walidacji.
- **Kalibracja** prawdopodobieństw po resamplingu / ważeniu.

#### Kluczowe zasady

- Resampling **tylko na zbiorze treningowym** i wewnątrz walidacji krzyżowej (pipeline `imblearn.pipeline.Pipeline`), inaczej leakage.
- Zbiór testowy zachowuje naturalny rozkład.
- Zacznij od prostych rzeczy (wagi klas + próg) zanim sięgniesz po SMOTE; często wystarcza.

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline([("smote", SMOTE(random_state=0)),
                 ("clf", RandomForestClassifier(class_weight="balanced"))])
```

**Źródła:**
- [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

---

<a id="q219"></a>
### 219. Jakie są typowe metryki ewaluacji w regresji?

**Odpowiedź:**

Niech $y_i$ to wartość rzeczywista, $\hat y_i$ predykcja, $n$ liczba próbek.

| Metryka | Wzór | Uwagi |
|---|---|---|
| MAE | $\frac1n\sum\lvert y_i-\hat y_i\rvert$ | średni błąd bezwzględny, w jednostkach celu, odporny na outliery |
| MSE | $\frac1n\sum(y_i-\hat y_i)^2$ | silnie karze duże błędy, gładki (dobry do optymalizacji) |
| RMSE | $\sqrt{MSE}$ | w jednostkach celu, wrażliwy na outliery |
| MAPE | $\frac{100}{n}\sum\left\lvert\frac{y_i-\hat y_i}{y_i}\right\rvert$ | względny; niestabilny dla $y\approx0$, asymetryczny |
| sMAPE / WAPE | warianty względne | stabilniejsze w szeregach czasowych |
| $R^2$ | $1-\frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$ | odsetek wyjaśnionej wariancji; może być ujemne |
| Adjusted $R^2$ | korekta o liczbę cech | karze za zbędne cechy |
| RMSLE | RMSE na $\log(1+y)$ | błędy względne, dla rozkładów skośnych |
| Huber loss | kwadratowa blisko 0, liniowa dalej | kompromis MAE/MSE |
| Quantile (pinball) loss | asymetryczna | predykcja kwantyli, przedziały |
| Median AE | mediana błędów | bardzo odporna |

#### Wskazówki

- Wybierz metrykę zgodną z kosztem biznesowym: gdy duże błędy są nieproporcjonalnie kosztowne, MSE/RMSE; gdy outliery to szum, MAE.
- $R^2$ mierzone na zbiorze testowym, nie treningowym, i nie porównuj $R^2$ między różnymi zbiorami.
- Przy szeregach czasowych: MASE (skalowany względem naiwnej prognozy), walidacja z rolling origin.
- Analizuj **residua** (wykresy residuów, heteroskedastyczność, autokorelacja), a nie tylko jedną liczbę.
- Raportuj baseline (średnia, mediana, prognoza naiwna).

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y, p)
rmse = mean_squared_error(y, p) ** 0.5
r2 = r2_score(y, p)
```

**Źródła:**
- [scikit-learn: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)
- [Coefficient of determination (Wikipedia)](https://en.wikipedia.org/wiki/Coefficient_of_determination)
- [Mean absolute percentage error (Wikipedia)](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)

---

<a id="q220"></a>
### 220. Jaka jest różnica między MAE, MSE i RMSE?

**Odpowiedź:**

Wszystkie trzy mierzą przeciętną wielkość błędu predykcji w regresji, ale różnią się sposobem karania błędów.

- **MAE** (Mean Absolute Error): $\frac1n\sum|e_i|$, gdzie $e_i=y_i-\hat y_i$.
- **MSE** (Mean Squared Error): $\frac1n\sum e_i^2$.
- **RMSE**: $\sqrt{\text{MSE}}$.

#### Różnice

| Aspekt | MAE | MSE | RMSE |
|---|---|---|---|
| Jednostka | jednostka celu | kwadrat jednostki | jednostka celu |
| Wrażliwość na outliery | mała (liniowa kara) | duża (kwadratowa) | duża |
| Estymator optymalny | **mediana** warunkowa | **średnia** warunkowa | średnia |
| Różniczkowalność | nieciągła pochodna w 0 | gładka | gładka |
| Interpretacja | „średni błąd" | wygodne matematycznie | „typowy błąd" z naciskiem na duże |

Zawsze $\text{MAE}\le\text{RMSE}\le\sqrt n\cdot\text{MAE}$. Duża różnica RMSE względem MAE sygnalizuje obecność dużych, rzadkich błędów (heavy tails).

#### Przykład liczbowy

Błędy: 1, 1, 1, 1, 10.
- MAE = 14/5 = 2,8.
- MSE = (4+100)/5 = 20,8, RMSE ≈ 4,56.
Jeden outlier ciągnie RMSE znacznie wyżej niż MAE.

#### Kiedy co

- **MAE**: gdy koszt rośnie liniowo z błędem, dane mają outliery, chcesz łatwej interpretacji.
- **MSE**: jako funkcja straty przy trenowaniu (gładkie gradienty, MLE przy szumie gaussowskim); duże błędy są szczególnie niepożądane.
- **RMSE**: do raportowania w jednostkach celu, gdy kwadratowa kara jest zamierzona.
- Alternatywa: **Huber loss**, gdy potrzebny kompromis.

Uwaga: trening na MSE, a ewaluacja w MAE (lub odwrotnie) może dawać inny ranking modeli; dopasuj stratę treningową do metryki docelowej.

**Źródła:**
- [scikit-learn: Mean absolute error](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error)
- [scikit-learn: Mean squared error](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error)
- [Root-mean-square deviation (Wikipedia)](https://en.wikipedia.org/wiki/Root-mean-square_deviation)

---

<a id="q221"></a>
### 221. Jak wybrać właściwą metrykę ewaluacji dla danego problemu?

**Odpowiedź:**

Metryka powinna wynikać z **celu biznesowego i kosztów błędów**, a nie z przyzwyczajenia. Proponowany proces:

#### 1. Zdefiniuj cel i koszty

- Co decyduje o sukcesie (przychód, bezpieczeństwo, satysfakcja użytkownika)?
- Jaki jest koszt FP vs FN? Czy błędy są symetryczne?
- Czy decyzja jest twarda (klasa) czy miękka (ranking, prawdopodobieństwo)?

#### 2. Rozpoznaj typ zadania i dane

| Zadanie | Typowe metryki |
|---|---|
| Klasyfikacja zbalansowana | accuracy, macro F1, log loss |
| Klasyfikacja niezbalansowana | PR AUC, recall@precision, F1, MCC |
| Ranking / rekomendacje | NDCG, MAP, MRR, recall@k |
| Regresja | MAE, RMSE, $R^2$, MAPE/MASE |
| Klasteryzacja | silhouette, ARI, NMI |
| Generowanie tekstu / LLM | BLEU/ROUGE (ograniczone), LLM-as-a-judge, ewaluacja ludzka |
| Detekcja obiektów | mAP@IoU |
| Prognozowanie | MASE, sMAPE, pinball loss |

#### 3. Wymagania dodatkowe

- Kalibracja prawdopodobieństw, fairness (metryki per grupa), odporność, latencja i koszt.
- Stabilność metryki na małych próbach, przedziały ufności.

#### 4. Metryka offline vs online

- Metryka offline (np. AUC) to proxy; ostatecznie weryfikuj metryki produktowe w **A/B teście** (CTR, retencja, przychód).
- Ustal **metrykę główną** (jedną, do optymalizacji) oraz **metryki ograniczające (guardrails)** (latencja, koszt, fairness).

#### Pułapki

- Optymalizowanie metryki, która nie odpowiada celowi (prawo Goodharta).
- Nadmierne dopasowanie do zbioru walidacyjnego.
- Ignorowanie dryfu rozkładu w produkcji.
- Raportowanie jednej liczby bez baseline'u.

**Źródła:**
- [scikit-learn: Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Goodhart's law (Wikipedia)](https://en.wikipedia.org/wiki/Goodhart%27s_law)
- [Google, Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)

---

<a id="q222"></a>
### 222. Jak porównać wydajność różnych modeli?

**Odpowiedź:**

Uczciwe porównanie modeli wymaga jednakowych warunków, właściwej metryki i oceny, czy różnica jest **statystycznie i praktycznie istotna**.

#### Procedura

1. **Ten sam podział danych** i ten sam protokół (te same foldy, ten sam zbiór testowy, te same seedy lub wiele seedów).
2. **Baseline**: model trywialny (klasa większościowa, średnia) i prosty model (regresja logistyczna).
3. **Walidacja krzyżowa** (stratified / grouped / time-series) zamiast pojedynczego podziału; raportuj średnią i odchylenie standardowe.
4. **Strojenie hiperparametrów** dla każdego modelu w podobnym budżecie (nested CV, by uniknąć optymistycznego biasu).
5. **Zbiór testowy** używany raz na końcu.
6. Metryki zgodne z celem plus koszty: latencja, pamięć, koszt inferencji, interpretowalność, łatwość utrzymania.

#### Testy istotności

- **Sparowany test t** na wynikach z foldów (uwaga na zależność foldów; poprawka Nadeau-Bengio), test Wilcoxona.
- **McNemar** dla dwóch klasyfikatorów na tym samym zbiorze testowym.
- **Bootstrap** przedziałów ufności różnicy metryki.
- Przy wielu modelach: test Friedmana + post-hoc (Nemenyi), poprawki na wielokrotne porównania.

#### Analiza dodatkowa

- Wyniki per segment/klasa (model może wygrywać średnio, ale przegrywać w kluczowym segmencie).
- Krzywe uczenia, kalibracja, odporność na dryf.
- Stabilność względem seeda.

#### Pułapki

- Data leakage, różne preprocessing dla modeli.
- „Wygrana" o 0,2 pp mieszcząca się w szumie.
- Porównanie strojonego modelu z niestrojonym.
- Wybór modelu na zbiorze testowym (test set reuse).
- W produkcji: ostateczne rozstrzygnięcie przez shadow deployment lub A/B test.

**Źródła:**
- [scikit-learn: Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Statistical Comparisons of Classifiers over Multiple Data Sets (Demšar, JMLR 2006)](https://jmlr.org/papers/v7/demsar06a.html)
- [scikit-learn: Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)

---

<a id="q223"></a>
### 223. Wyjaśnij walidację krzyżową (cross-validation) i jej znaczenie.

**Odpowiedź:**

**Cross-validation (CV)** to procedura szacowania zdolności generalizacji modelu przez wielokrotny podział danych na część treningową i walidacyjną. Pojedynczy podział train/test bywa niestabilny (zależy od szczęścia w losowaniu) i marnuje dane; CV wykorzystuje każdą próbkę zarówno do trenowania, jak i walidacji.

#### K-fold

1. Podziel dane na $k$ równych foldów (zwykle $k=5$ lub $10$).
2. Dla $i=1..k$: trenuj na $k-1$ foldach, oceń na $i$-tym.
3. Uśrednij metrykę: $\text{CV}=\frac1k\sum_i m_i$ i podaj odchylenie.

#### Warianty

- **Stratified k-fold**: zachowuje proporcje klas (klasyfikacja, imbalance).
- **Leave-One-Out (LOO)**: $k=n$; mały bias, duża wariancja i koszt.
- **Repeated k-fold**: stabilniejsze oszacowanie.
- **Group k-fold**: próbki tej samej grupy (pacjent, użytkownik) nie mogą być w train i val jednocześnie.
- **TimeSeriesSplit** / rolling origin: dane czasowe, walidacja tylko „w przód".
- **Nested CV**: pętla wewnętrzna do strojenia, zewnętrzna do oceny.

#### Po co

- Wiarygodniejsze oszacowanie błędu generalizacji i jego zmienności.
- Wybór modelu i hiperparametrów (`GridSearchCV`, `RandomizedSearchCV`).
- Wykrywanie overfittingu i niestabilności modelu.
- Szczególnie cenne przy małych zbiorach.

#### Kompromis bias-wariancja wyboru $k$

Mały $k$: większy pesymistyczny bias (mniej danych do treningu), mniejsza wariancja i koszt. Duży $k$: mniejszy bias, większy koszt.

#### Pułapki

- **Leakage**: skalowanie, imputacja, selekcja cech i resampling muszą być wewnątrz pipeline w każdym foldzie.
- Losowy KFold na danych czasowych lub grupowych zawyża wyniki.
- Strojenie na CV i raportowanie tego samego CV jest optymistyczne (użyj nested CV lub osobnego testu).
- Przy bardzo dużych danych lub kosztownych modelach (deep learning) zwykle wystarcza jeden hold-out.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
pipe = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="f1")
```

**Źródła:**
- [scikit-learn: Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Understanding Cross-Validation in Machine Learning (Amit Shekhar, X)](https://x.com/amitiitbhu/status/1939929240084128137)
- [Cross-validation (Wikipedia)](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

---

<a id="q224"></a>
### 224. Czym jest strojenie hiperparametrów (Hyperparameter Tuning)?

**Odpowiedź:**

**Hiperparametry** to ustawienia modelu i procesu uczenia wybierane **przed** treningiem, a nie uczone z danych: learning rate, liczba drzew, głębokość drzewa, siła regularyzacji ($\lambda$, $C$), rozmiar batcha, liczba warstw, dropout, $k$ w kNN. **Parametry** (wagi) są natomiast optymalizowane przez algorytm uczący. **Hyperparameter tuning** to poszukiwanie konfiguracji hiperparametrów maksymalizującej metrykę na **zbiorze walidacyjnym** (lub w CV).

#### Metody

| Metoda | Opis | Uwagi |
|---|---|---|
| Grid search | pełna siatka kombinacji | prosty, koszt rośnie wykładniczo z liczbą wymiarów |
| Random search | losowe próbkowanie z rozkładów | zwykle wydajniejszy od grid, gdy tylko część hiperparametrów jest istotna (Bergstra & Bengio) |
| Optymalizacja bayesowska (TPE, GP) | model zastępczy wybiera kolejne punkty | efektywna przy drogich ewaluacjach (Optuna, Hyperopt) |
| Successive halving / Hyperband / ASHA | wczesne zatrzymywanie słabych konfiguracji | oszczędza budżet |
| Population-based training | ewolucyjne, na bieżąco | w RL i dużych sieciach |

#### Dobre praktyki

- Definiuj **przestrzeń poszukiwań**: skala logarytmiczna dla learning rate, $C$, $\lambda$.
- Używaj **CV** lub stałego zbioru walidacyjnego; **zbiór testowy nie bierze udziału** w strojeniu.
- Zaczynaj od szerokiego zakresu i zawężaj; strojenie najbardziej wpływowych parametrów najpierw (learning rate, regularyzacja).
- Early stopping, ustalone seedy, logowanie eksperymentów (MLflow, Weights & Biases).
- Rozważ koszt: czasem lepsze dane lub cechy dają więcej niż dalsze strojenie.

#### Pułapki

- Overfitting do zbioru walidacyjnego przy dużej liczbie prób; użyj nested CV.
- Leakage w pipeline.
- Ignorowanie wariancji metryki (różnice mniejsze niż szum).

```python
import optuna
def objective(trial):
    C = trial.suggest_float("C", 1e-3, 1e2, log=True)
    return cross_val_score(LogisticRegression(C=C), X, y, cv=5).mean()
study = optuna.create_study(direction="maximize"); study.optimize(objective, n_trials=50)
```

**Źródła:**
- [scikit-learn: Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
- [Random Search for Hyper-Parameter Optimization (Bergstra & Bengio)](https://jmlr.org/papers/v13/bergstra12a.html)
- [Optuna documentation](https://optuna.readthedocs.io/)
- [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560)

---

<a id="q225"></a>
### 225. Jak ocenia się modele uczenia nienadzorowanego?

**Odpowiedź:**

W uczeniu nienadzorowanym brak etykiet „prawdy", więc ewaluacja jest trudniejsza i zwykle wielowymiarowa. Stosuje się trzy grupy podejść.

#### 1. Metryki wewnętrzne (intrinsic)

Bazują wyłącznie na danych i wynikach modelu:
- **Klasteryzacja**: silhouette, Davies-Bouldin, Calinski-Harabasz, inertia (elbow), stabilność klastrów.
- **Redukcja wymiarowości**: wyjaśniona wariancja (PCA), **błąd rekonstrukcji** (autoenkodery, PCA), zachowanie sąsiedztw (trustworthiness, continuity), dla t-SNE/UMAP także wizualna inspekcja (ostrożnie).
- **Modele generatywne / gęstości**: log-likelihood na zbiorze walidacyjnym, perplexity, dla GAN/dyfuzji FID, Inception Score, precision/recall rozkładów.
- **Wykrywanie anomalii**: bez etykiet np. stabilność, spójność z regułami; z małą liczbą etykiet AUC/PR AUC.
- **Reprezentacje / self-supervised**: **linear probing** (klasyfikator liniowy na zamrożonych embeddingach), kNN accuracy.

#### 2. Metryki zewnętrzne (extrinsic)

Gdy istnieje choćby częściowa prawda odniesienia: ARI, NMI, homogeneity/completeness, V-measure, Fowlkes-Mallows.

#### 3. Ocena pośrednia i biznesowa

- **Downstream task**: czy klastry lub embeddingi poprawiają model nadzorowany, wyszukiwanie, rekomendacje?
- **Ocena ekspercka**: interpretowalność, sensowność segmentów.
- **Stabilność**: powtarzalność wyników przy bootstrapie i różnych seedach.
- **A/B test** dla efektu produktowego.

#### Wskazówki

- Nie polegaj na jednej metryce; łącz metryki i wizualizację.
- Dobór liczby klastrów $k$: elbow, silhouette, gap statistic, kryteria informacyjne (BIC/AIC) dla GMM.
- Ważna jest skala cech i miara odległości, które determinują wynik.

**Źródła:**
- [scikit-learn: Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- [scikit-learn: Selecting the number of clusters with silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
- [GANs Trained by a Two Time-Scale Update Rule (FID)](https://arxiv.org/abs/1706.08500)

---

<a id="q226"></a>
### 226. Jak ocenić algorytm klasteryzacji?

**Odpowiedź:**

Ocena klasteryzacji zależy od tego, czy dysponujemy etykietami odniesienia.

#### Metryki wewnętrzne (bez etykiet)

- **Silhouette coefficient**: dla punktu $i$, $s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}$, gdzie $a$ to średnia odległość do własnego klastra, $b$ do najbliższego innego klastra. Zakres $[-1,1]$; im bliżej 1, tym lepiej rozdzielone. Preferuje klastry wypukłe.
- **Davies-Bouldin index**: średnie podobieństwo klastra do najbardziej podobnego; niższy = lepszy.
- **Calinski-Harabasz** (Variance Ratio): stosunek rozproszenia między klastrami do wewnątrz klastrów; wyższy = lepszy.
- **Inertia / WCSS** i metoda łokcia (elbow) dla k-means.
- **Dunn index**, **DBCV** dla klastrów o dowolnym kształcie (DBSCAN).
- Dla GMM: **BIC / AIC**, log-likelihood.

#### Metryki zewnętrzne (z etykietami)

- **Adjusted Rand Index (ARI)**: zgodność par, skorygowana o przypadek; ~0 dla losowego, 1 dla idealnego.
- **Normalized Mutual Information (NMI)**, **AMI**.
- **Homogeneity, completeness, V-measure**.
- **Fowlkes-Mallows**, purity (prosta, ale faworyzuje wiele małych klastrów).

#### Ocena stabilności i użyteczności

- **Stabilność**: powtórz klasteryzację na próbkach bootstrap / z innym seedem i porównaj ARI.
- Wizualizacja (PCA/UMAP), profilowanie klastrów (średnie cech), interpretacja domenowa.
- Efekt na zadanie docelowe (segmentacja klientów, kompresja, feature dla modelu).

#### Pułapki

- Metryki wewnętrzne są uprzedzone do kształtu klastrów (silhouette lubi kuliste); DBSCAN z szumem może wyglądać słabo.
- Skalowanie cech i wybór odległości silnie wpływają na wynik.
- Metryki wewnętrzne mogą rosnąć dla „ładnych", lecz bezużytecznych biznesowo klastrów.

```python
from sklearn.metrics import silhouette_score, adjusted_rand_score
sil = silhouette_score(X, labels)
ari = adjusted_rand_score(y_true, labels)  # jeśli są etykiety
```

**Źródła:**
- [scikit-learn: Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)
- [Silhouette (clustering) (Wikipedia)](https://en.wikipedia.org/wiki/Silhouette_(clustering))
- [Rand index (Wikipedia)](https://en.wikipedia.org/wiki/Rand_index)

---

<a id="q227"></a>
### 227. Jakich metryk użyjesz dla systemu rekomendacyjnego?

**Odpowiedź:**

Ewaluacja rekomendera ma trzy warstwy: metryki predykcji ocen, metryki rankingowe (top-k) oraz metryki „poza trafnością" (beyond accuracy), a ostatecznie metryki online.

#### Predykcja ocen (explicit feedback)

- **RMSE / MAE** na ocenach (dziś rzadziej, bo użytkownik widzi listę, a nie ocenę).

#### Metryki rankingowe (top-k)

- **Precision@k**: odsetek trafnych wśród $k$ rekomendacji.
- **Recall@k**: odsetek trafnych elementów użytkownika odnalezionych w top-$k$.
- **MAP** (Mean Average Precision), **MRR** (Mean Reciprocal Rank: $1/\text{rank}$ pierwszego trafnego).
- **NDCG@k**: uwzględnia pozycję i gradację trafności:
$$DCG@k=\sum_{i=1}^k\frac{2^{rel_i}-1}{\log_2(i+1)},\quad NDCG@k=\frac{DCG@k}{IDCG@k}$$
- **Hit rate@k**, **AUC** (para pozytyw-negatyw).

#### Beyond accuracy

- **Coverage** (jaka część katalogu jest rekomendowana), **diversity**, **novelty**, **serendipity**, **popularity bias**, **fairness** wobec dostawców/grup.
- **Kalibracja** (dopasowanie do zainteresowań użytkownika).

#### Metryki online

- CTR, konwersja, czas oglądania, retencja, przychód, ostatecznie **A/B test**. Metryki offline często słabo korelują z online.

#### Dobre praktyki

- **Podział czasowy** (train na przeszłości, test na przyszłości) zamiast losowego, aby uniknąć leakage.
- Uwaga na **bias selekcji** (widzimy tylko to, co wcześniej rekomendowano); rozważ off-policy evaluation, IPS, randomizowany ruch eksploracyjny.
- **Negative sampling** przy ewaluacji wpływa na wyniki; przeprowadzaj ewaluację na pełnym rankingu, gdy to możliwe.
- Uwzględnij **cold start** (nowi użytkownicy/przedmioty) osobno.
- Guardrails: latencja, różnorodność, treści niepożądane.

**Źródła:**
- [Evaluation measures (information retrieval) (Wikipedia)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [Evaluating Recommendation Systems (Shani & Gunawardana)](https://link.springer.com/chapter/10.1007/978-0-387-85820-3_8)
- [scikit-learn: ndcg_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)

---

<a id="q228"></a>
### 228. Czym jest A/B testing w kontekście ML?

**Odpowiedź:**

**A/B test** to kontrolowany eksperyment online, w którym użytkownicy są losowo przydzielani do grupy **kontrolnej (A)**, korzystającej z obecnego modelu, i **testowej (B)**, korzystającej z nowego modelu. Porównujemy metryki produktowe, aby ustalić, czy nowy model **przyczynowo** poprawia wynik. Metryki offline (AUC, NDCG) są tylko przybliżeniem; A/B test mierzy rzeczywisty efekt (CTR, konwersja, retencja, przychód).

#### Etapy

1. **Hipoteza i metryki**: metryka główna (OEC), metryki pomocnicze i **guardrails** (latencja, błędy, koszt, fairness).
2. **Randomizacja**: przydział na poziomie użytkownika (stabilne hashowanie ID), aby doświadczenie było spójne.
3. **Moc statystyczna i rozmiar próby**: na podstawie minimalnego wykrywalnego efektu (MDE), wariancji, poziomu istotności $\alpha$ (zwykle 0,05) i mocy $1-\beta$ (zwykle 0,8).
4. **Czas trwania**: pełne cykle tygodniowe, aby uchwycić sezonowość; bez „podglądania" wyników.
5. **Analiza**: test t / z dla proporcji, bootstrap, przedziały ufności, **poprawki na wielokrotne testy**.
6. **Decyzja**: istotność statystyczna oraz praktyczna.

#### Specyfika ML

- **Shadow deployment** i **canary** przed A/B: bezpieczne sprawdzenie modelu.
- **Interleaving** w rankingu: czulszy niż A/B przy porównaniu rankerów.
- **Multi-armed bandits**: adaptacyjny przydział ruchu.
- **Efekty sieciowe / interferencja** (marketplace, media społecznościowe): użytkownicy wpływają na siebie, potrzebne randomizacje klastrowe.
- **Novelty effect** i **efekt sprzężenia zwrotnego**: model wpływa na dane, na których będzie trenowany.
- **Sample Ratio Mismatch (SRM)**: kontrola, czy podział ruchu odpowiada założonemu; sygnał błędu w implementacji.
- **CUPED** zmniejsza wariancję dzięki danym sprzed eksperymentu.

#### Pułapki

- Peeking i wczesne zatrzymanie, wiele metryk bez korekty (p-hacking).
- Za krótki test, za mała próba.
- Niedopasowanie jednostki randomizacji do jednostki analizy.

```python
from statsmodels.stats.proportion import proportions_ztest
stat, p = proportions_ztest([conv_b, conv_a], [n_b, n_a])
```

**Źródła:**
- [A/B testing (Wikipedia)](https://en.wikipedia.org/wiki/A/B_testing)
- [Trustworthy Online Controlled Experiments (Kohavi, Tang, Xu)](https://experimentguide.com/)
- [statsmodels: proportions_ztest](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.proportions_ztest.html)

---

<a id="q229"></a>
### 229. Czym jest LLM as a Judge?

**Odpowiedź:**

**LLM-as-a-Judge** to podejście, w którym mocny model językowy pełni rolę **sędziego** oceniającego odpowiedzi innych modeli (lub systemów, np. RAG, agentów) zamiast lub obok annotatorów ludzkich. Metryki takie jak BLEU/ROUGE słabo korelują z jakością otwartego tekstu, a ocena ludzka jest droga i wolna; LLM-judge jest tanim, skalowalnym przybliżeniem.

#### Warianty

- **Pointwise / rubric-based**: sędzia ocenia pojedynczą odpowiedź według kryteriów (np. 1-5 za trafność, spójność, zgodność z kontekstem). Przykład: G-Eval.
- **Pairwise**: wybiera lepszą z dwóch odpowiedzi (używane w MT-Bench, Chatbot Arena, do preferencji dla RLHF/DPO).
- **Reference-based**: porównanie z odpowiedzią wzorcową.
- **Reference-free**: ocena jakości, bezpieczeństwa, faithfulness względem kontekstu (RAG: Ragas).

#### Implementacja

```python
prompt = f"""Oceń odpowiedź według rubryki (1-5): poprawność, kompletność, zwięzłość.
Pytanie: {q}
Odpowiedź: {a}
Zwróć JSON: {{"reasoning": "...", "score": int}}"""
```

Dobre praktyki: jasna rubryka z opisem poziomów, **rozumowanie przed oceną** (chain-of-thought), temperatura 0, ustrukturyzowane wyjście, przykłady few-shot, kilka ocen i uśrednienie.

#### Znane błędy sędziów (biases)

- **Position bias**: preferencja pierwszej lub drugiej odpowiedzi; mitigacja: zamiana kolejności i uśrednienie.
- **Verbosity bias**: dłuższe odpowiedzi oceniane wyżej.
- **Self-preference / self-enhancement**: model faworyzuje własny styl.
- Słabość w zadaniach wymagających ścisłej weryfikacji (matematyka, kod) bez referencji.
- Wrażliwość na sformułowanie promptu, niestabilność wersji modelu.

#### Walidacja

- **Skalibruj sędziego względem ludzi** na zbiorze złotych etykiet (zgodność, np. Cohen's kappa, korelacja Spearmana). W MT-Bench zgodność GPT-4 z ludźmi była na poziomie porównywalnym do zgodności między ludźmi (ok. 80%).
- Używaj sędziego innego niż oceniany model, monitoruj dryf.
- Traktuj ocenę jako sygnał uzupełniający testy deterministyczne i ocenę ludzką dla przypadków krytycznych.

**Źródła:**
- [LLM as a Judge (Outcome School)](https://outcomeschool.com/blog/llm-as-a-judge)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)
- [Ragas documentation](https://docs.ragas.io/)

---

## Projektowanie systemów i MLOps

<a id="q230"></a>
### 230. Zaprojektuj agenta głosowego AI działającego w czasie rzeczywistym (Real-Time Voice AI Agent)

**Odpowiedź:**

#### Wymagania

- Funkcjonalne: rozmowa głosowa w obie strony, obsługa przerwań (**barge-in**), narzędzia (kalendarz, CRM), wiele języków.
- Niefunkcjonalne: opóźnienie odpowiedzi „mouth-to-ear" ok. 500-800 ms (rozmowa staje się nienaturalna powyżej ~1 s), skalowalność (tysiące współbieżnych sesji), niezawodność, prywatność.

#### Architektura kaskadowa

```
Mikrofon -> [WebRTC/WebSocket] -> VAD/turn detection -> streaming ASR
   -> LLM (streaming, tool calls) -> streaming TTS -> [WebRTC] -> Głośnik
```

1. **Transport**: WebRTC (UDP, niskie opóźnienie, echo cancellation, jitter buffer) dla przeglądarki/mobile; SIP/PSTN dla telefonii.
2. **VAD i wykrywanie końca wypowiedzi** (endpointing): Silero VAD, semantyczne turn detection; kompromis: zbyt szybkie ucięcie vs zbyt długie oczekiwanie.
3. **ASR streaming** (np. Whisper w wariancie streamingowym, modele typu Conformer/RNN-T): częściowe transkrypcje.
4. **LLM**: streamowanie tokenów; krótkie odpowiedzi, mały/szybki model lub speculative decoding, prefix caching promptu systemowego; function calling z „wypełniaczami" („już sprawdzam") maskującymi opóźnienie narzędzi.
5. **TTS streaming**: synteza zdanie po zdaniu / chunk po chunku; pierwszy dźwięk zanim LLM skończy.
6. **Barge-in**: gdy użytkownik zaczyna mówić, natychmiast przerwij TTS i generację, odrzuć nieodtworzony bufor i zaktualizuj kontekst o faktycznie wypowiedziany fragment.

#### Alternatywa: speech-to-speech

Modele end-to-end (audio in / audio out) redukują opóźnienie i zachowują prozodię/emocje, ale trudniej kontrolować i debugować; podejście hybrydowe bywa kompromisem.

#### Budżet opóźnień (przykład orientacyjny)

Sieć ~50-100 ms, endpointing ~200 ms, ASR końcowy ~100 ms, TTFT LLM ~200-300 ms, pierwszy chunk TTS ~100-200 ms. Każdy etap musi być streamowany i pipeline'owany.

#### Skalowanie i operacje

- Sesje stanowe: sticky routing, autoscaling GPU dla ASR/TTS/LLM, batching ciągły.
- Regiony blisko użytkownika, monitoring p50/p95 opóźnień per etap, WER, przerwania, task success.
- Bezpieczeństwo: PII redaction, zgody na nagrywanie, ograniczenie prompt injection przez narzędzia.
- Ewaluacja: symulowane rozmowy, LLM-as-judge, testy audio z akcentami i szumem.

**Źródła:**
- [Design a Real-Time Voice AI Agent (Outcome School)](https://outcomeschool.com/blog/design-a-real-time-voice-ai-agent)
- [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356)
- [Silero VAD (GitHub)](https://github.com/snakers4/silero-vad)
- [WebRTC](https://webrtc.org/)

---

<a id="q231"></a>
### 231. Zaprojektuj ChatGPT: od treningu do serwowania (end to end)

**Odpowiedź:**

#### Wymagania

Asystent czatowy dla milionów użytkowników: jakościowy, bezpieczny, z niskim opóźnieniem (streaming), z pamięcią rozmowy, tanio w utrzymaniu.

#### Faza 1: dane i pretraining

- Zbieranie i czyszczenie korpusu (web, książki, kod): deduplikacja, filtrowanie jakości i toksyczności, usuwanie PII, kontrola licencji.
- Tokenizacja (BPE), mieszanka danych, ewaluacja kontaminacji benchmarków.
- Trening dekoderowego Transformera z next-token prediction; **skalowanie** według praw skalowania (Chinchilla: dopasowanie liczby parametrów i tokenów do budżetu obliczeń).
- Równoległość: data / tensor / pipeline / sequence parallelism, ZeRO/FSDP, mixed precision (bf16), checkpointing, odporność na awarie sprzętu.

#### Faza 2: post-training (alignment)

1. **SFT** na demonstracjach instrukcji.
2. **Preferencje**: model nagrody (RLHF/PPO) lub DPO; RLAIF i Constitutional AI dla skalowania; RL z weryfikowalnymi nagrodami dla rozumowania.
3. Red-teaming, ewaluacje bezpieczeństwa, ewaluacja zdolności (benchmarki, LLM-as-judge, ocena ludzka).

#### Faza 3: serwowanie

- **Optymalizacja inferencji**: kwantyzacja (INT8/FP8/INT4), KV cache, PagedAttention, continuous batching, FlashAttention, speculative decoding, GQA, prefix caching.
- **Równoległość modelu** (tensor parallel) na wielu GPU, rozdzielenie prefill/decode.
- **Routing i kolejki**, autoscaling, limity zapytań, priorytety, obsługa wielu modeli (małe do prostych zapytań, duże do trudnych).
- **Streaming** (SSE/WebSocket) tokenów.

#### Warstwa aplikacyjna

- Zarządzanie kontekstem: historia, skracanie/streszczanie, pamięć długoterminowa.
- Narzędzia: wyszukiwarka, interpreter kodu, RAG, function calling.
- **Moderacja** wejścia i wyjścia (klasyfikatory bezpieczeństwa), ochrona przed prompt injection.
- Telemetria: zgoda na użycie rozmów do treningu, feedback (kciuk w górę/w dół) zasilający kolejne iteracje.

#### MLOps i ewaluacja

Wersjonowanie danych/modeli, CI dla ewaluacji, shadow/canary/A-B rollout, monitoring jakości (LLM-judge na próbce ruchu), koszt na token, opóźnienia TTFT/TPOT, dryf zapytań, plan rollbacku.

**Źródła:**
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290)

---

<a id="q232"></a>
### 232. Zaprojektuj system RAG (rozmowa z własnymi dokumentami)

**Odpowiedź:**

**RAG** (Retrieval-Augmented Generation) łączy wyszukiwanie w zewnętrznej bazie wiedzy z generowaniem przez LLM, co ogranicza halucynacje i pozwala odpowiadać na podstawie aktualnych, prywatnych danych bez fine-tuningu.

#### Wymagania

Odpowiedzi ugruntowane w dokumentach z cytatami, kontrola dostępu per użytkownik, aktualizacje dokumentów, opóźnienie kilku sekund, skala np. miliony fragmentów.

#### Potok indeksowania (offline)

1. **Ingestion**: konektory (PDF, HTML, Confluence, Drive), parsowanie z zachowaniem struktury (tabele, nagłówki), OCR.
2. **Chunking**: 200-800 tokenów, z overlapem, po granicach semantycznych / nagłówkach; metadane (źródło, strona, data, uprawnienia).
3. **Embedding** modelem osadzeń; zapis w **wektorowej bazie** (FAISS, pgvector, Qdrant, Milvus, OpenSearch) z indeksem ANN (HNSW / IVF-PQ) plus indeks leksykalny (BM25).
4. Inkrementalne aktualizacje, wersjonowanie i usuwanie.

#### Potok zapytania (online)

1. **Przepisanie zapytania** (query rewriting, uwzględnienie historii, HyDE, multi-query).
2. **Hybrid retrieval**: dense + BM25, filtry po metadanych i uprawnieniach (ACL egzekwowane w retrievalu).
3. **Reranking** cross-encoderem (top-50 do top-5/10).
4. **Budowa promptu**: fragmenty z identyfikatorami, instrukcja „odpowiadaj tylko na podstawie kontekstu, cytuj źródła, powiedz, gdy nie wiesz".
5. **Generacja** ze streamingiem, weryfikacja cytatów (faithfulness).

#### Ewaluacja

- Retrieval: recall@k, MRR, NDCG na zbiorze pytań z oznaczonymi fragmentami.
- Generacja: faithfulness / groundedness, answer relevance, correctness (LLM-as-judge, Ragas), ocena ludzka.
- Metryki online: feedback, odsetek „nie wiem", koszty i latencja.

#### Zaawansowane

Parent-document retrieval, GraphRAG, agentic RAG (iteracyjne wyszukiwanie), cache semantyczny, multimodalne dokumenty (ColPali), Self-RAG.

#### Pułapki

Zły chunking i parsowanie tabel, brak ACL w retrievalu (wyciek danych), prompt injection z dokumentów, „lost in the middle", przestarzałe indeksy, zbyt wiele fragmentów w kontekście.

**Źródła:**
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- [Efficient and robust ANN using HNSW](https://arxiv.org/abs/1603.09320)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Ragas documentation](https://docs.ragas.io/)

---

<a id="q233"></a>
### 233. Zaprojektuj pamięć dla osobistego asystenta AI

**Odpowiedź:**

LLM jest bezstanowy; „pamięć" to system zewnętrzny decydujący, **co zapisać, jak przechowywać, jak odzyskać i kiedy zapomnieć**.

#### Rodzaje pamięci

| Typ | Zawartość | Realizacja |
|---|---|---|
| Krótkoterminowa (working) | bieżąca rozmowa | okno kontekstu, streszczanie starszych tur |
| Epizodyczna | zdarzenia, wcześniejsze sesje | log rozmów + embeddingi |
| Semantyczna | fakty i preferencje użytkownika (imię, alergie, styl) | magazyn faktów (key-value / graf wiedzy) |
| Proceduralna | jak wykonywać zadania, nauczone workflow | instrukcje, reguły, skille |

#### Architektura

1. **Zapis (write path)**: po turze (asynchronicznie) mały model ekstrahuje kandydatów na wspomnienia, ocenia ich ważność, deduplikuje i **konsoliduje** (aktualizacja sprzecznych faktów, np. „przeprowadził się do Gdańska"), zapisuje z metadanymi (czas, źródło, pewność).
2. **Odczyt (read path)**: zapytanie użytkownika -> retrieval hybrydowy (wektorowy + słowa kluczowe + filtry) -> ranking po trafności, świeżości i ważności (jak w *Generative Agents*) -> wstrzyknięcie do promptu w ramach budżetu tokenów.
3. **Zarządzanie kontekstem**: podejście MemGPT: model sam wywołuje narzędzia do odczytu/zapisu pamięci (hierarchia jak pamięć operacyjna/dysk).
4. **Zapominanie i konsolidacja**: decay, streszczanie, usuwanie na żądanie (RODO: prawo do usunięcia).

#### Prywatność i bezpieczeństwo

- Szyfrowanie, izolacja per użytkownik, przejrzystość (użytkownik widzi i edytuje wspomnienia), zgody, retencja.
- Ochrona przed **memory poisoning** i prompt injection zapisującym fałszywe „wspomnienia".
- Unikaj zapamiętywania danych wrażliwych bez zgody.

#### Ewaluacja

Testy pamięci długoterminowej (odtwarzanie faktów po wielu sesjach, aktualizacje, odporność na sprzeczności), precision/recall retrievalu wspomnień, wpływ na satysfakcję i koszt tokenów, odsetek szkodliwych/nietrafnych wstrzyknięć.

**Źródła:**
- [AI Agent Memory (Outcome School)](https://outcomeschool.com/blog/ai-agent-memory)
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)

---

<a id="q234"></a>
### 234. Zaprojektuj agenta Deep Research

**Odpowiedź:**

**Deep Research agent** to system, który dla złożonego pytania samodzielnie planuje, wielokrotnie przeszukuje źródła, czyta je, syntetyzuje i produkuje raport z cytowaniami; zadanie trwa minuty, nie sekundy.

#### Wymagania

Wysoka trafność i pokrycie, weryfikowalne cytowania, kontrola kosztu i czasu, odporność na błędy narzędzi, możliwość przerwania i wznowienia.

#### Architektura

1. **Planner**: rozbija pytanie na podpytania, tworzy plan badań (drzewo/lista zadań), pozwala użytkownikowi doprecyzować zakres.
2. **Pętla agentowa (ReAct)**: myśl -> wywołaj narzędzie -> obserwacja -> aktualizuj plan.
3. **Narzędzia**: wyszukiwarka webowa, pobieranie i parsowanie stron/PDF, wyszukiwanie w dokumentach prywatnych (RAG), interpreter kodu (analiza danych), kalkulator.
4. **Subagenci równoległi** (orchestrator-workers): każdy bada podtemat we własnym oknie kontekstu i zwraca zwięzłe streszczenie z dowodami; ogranicza to rozrost kontekstu głównego agenta.
5. **Zarządzanie kontekstem i pamięcią**: notatki robocze (scratchpad), kompresja/streszczanie obserwacji, zapis wyników pośrednich do magazynu.
6. **Synteza**: raport strukturalny z cytatami powiązanymi ze źródłami; osobny krok **weryfikacji** (sprawdzenie, czy każde twierdzenie ma wsparcie w źródle; wykrycie sprzeczności między źródłami).
7. **Kryteria stopu**: pokrycie podpytań, malejące nowe informacje, budżet czasu/tokenów.

#### Jakość i ewaluacja

- Ocena raportu rubryką (LLM-as-judge + eksperci): kompletność, poprawność, jakość źródeł, cytowania.
- Benchmarki wieloetapowego wyszukiwania (np. BrowseComp, GAIA) oraz własny zestaw zadań.
- Trening przez RL na nagrodach za poprawność końcową (podejście stosowane w nowszych agentach).

#### Ryzyka

- **Prompt injection** z odwiedzanych stron (izolacja treści z narzędzi, brak nadawania uprawnień treści pobranej).
- Halucynowane cytowania, ranking SEO-spamu, źródła niskiej wiarygodności (scoring domen).
- Koszty i pętle bez końca (limity kroków, budżet tokenów).
- Zgodność z robots.txt, prawa autorskie, prywatność.
- Obserwowalność: pełny trace kroków, powtarzalność, możliwość audytu.

**Źródła:**
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983)
- [Anthropic: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)

---

<a id="q235"></a>
### 235. Zaprojektuj wieloagentowy system obsługi klienta (Multi-Agent Customer Support)

**Odpowiedź:**

#### Wymagania

Obsługa czatu/e-maila/głosu, rozwiązywanie typowych spraw automatycznie (zwroty, status zamówienia, FAQ), eskalacja do człowieka, bezpieczeństwo akcji (płatności, dane osobowe), zgodność z polityką firmy, mierzalna jakość.

#### Architektura

- **Router / triage agent**: klasyfikuje intencję, język, sentyment, pilność; kieruje do wyspecjalizowanego agenta.
- **Agenci specjalistyczni**: FAQ/wiedza (RAG po bazie wiedzy), zamówienia i logistyka, płatności/zwroty, wsparcie techniczne, onboarding. Każdy ma własny prompt, narzędzia i zakres uprawnień (zasada najmniejszych uprawnień).
- **Supervisor / orchestrator** (wzorzec orchestrator-workers lub handoff): zarządza przekazaniem między agentami, współdzieli stan rozmowy.
- **Narzędzia (API)**: CRM, system zamówień, płatności, ticketing; wywołania przez function calling / MCP z walidacją schematu.
- **Guardrails**: filtry PII, kontrola polityk (limity zwrotów), weryfikacja tożsamości przed akcjami wrażliwymi, **human approval** dla operacji o wysokim ryzyku.
- **Eskalacja do człowieka**: przy niskiej pewności, frustracji klienta, prośbie klienta lub sprawach prawnych; przekazanie z podsumowaniem i historią.
- **Pamięć**: profil klienta i historia zgłoszeń (z zachowaniem prywatności).

#### Przepływ

Wiadomość -> triage -> agent domenowy (retrieval + narzędzia) -> odpowiedź weryfikowana przez guardrails -> wysłanie; asynchronicznie logowanie, feedback, aktualizacja ticketu.

#### Ewaluacja i monitoring

- **Resolution rate**, containment (bez człowieka), CSAT, średni czas obsługi, odsetek eskalacji, koszt na zgłoszenie.
- Jakość: LLM-as-judge na próbce, zgodność z polityką, poprawność wywołań narzędzi, testy regresji na zestawie scenariuszy (symulowani klienci).
- Ścieżki i trace'y każdego agenta, alerty na anomalie.

#### Pułapki wieloagentowości

Więcej agentów oznacza większy koszt, opóźnienie i trudniejsze debugowanie; zaczynaj od jednego agenta z narzędziami i dziel dopiero, gdy prompty/narzędzia stają się zbyt szerokie. Ryzyka: pętle przekazywań, gubienie kontekstu, propagacja błędów, prompt injection przez treść wiadomości klienta.

**Źródła:**
- [Multi-Agent Systems (Outcome School)](https://outcomeschool.com/blog/multi-agent-systems)
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://arxiv.org/abs/2308.08155)
- [Anthropic: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

---

<a id="q236"></a>
### 236. Zaprojektuj asystenta AI działającego na urządzeniu (On-Device AI Assistant)

**Odpowiedź:**

#### Motywacje i ograniczenia

- **Zalety**: prywatność (dane nie opuszczają urządzenia), niskie i przewidywalne opóźnienie, praca offline, brak kosztu serwera na zapytanie.
- **Ograniczenia**: pamięć RAM (kilka GB dla modelu), moc obliczeniowa, bateria i nagrzewanie, przepustowość pamięci (decode jest memory-bound), różnorodność sprzętu.

#### Model

- Mały model językowy (rzędu 1-4B parametrów) lub destylowany z większego (knowledge distillation).
- **Kwantyzacja**: INT8 / INT4 (GPTQ, AWQ, formaty GGUF w llama.cpp), także KV cache; kompromis jakość vs rozmiar (4-bit zmniejsza wagi ok. 4x względem FP16).
- Techniki efektywności: GQA, sliding window attention, speculative decoding z małym modelem szkicowym, pruning.
- **Adaptery LoRA** do personalizacji i wielu zadań przy współdzielonych wagach bazowych.

#### Runtime i sprzęt

- Silniki: llama.cpp, ExecuTorch, Core ML, TensorFlow Lite / LiteRT, ONNX Runtime, MLC LLM.
- Wykorzystanie **NPU/GPU** zamiast CPU; pamięć mapowana (mmap), ładowanie warstw na żądanie.
- Zarządzanie zasobami: ładowanie/zwalnianie modelu, tryby oszczędzania baterii, ograniczanie długości kontekstu.

#### Architektura hybrydowa

- **Routing lokalnie/chmura**: proste zapytania i wrażliwe dane lokalnie, złożone (długie rozumowanie, wiedza aktualna) do chmury za zgodą użytkownika; fallback przy braku łączności.
- Lokalny **RAG** po danych użytkownika (wiadomości, pliki) z lokalnym indeksem wektorowym i szyfrowanym magazynem.
- Narzędzia: intencje/API systemu operacyjnego (kalendarz, alarmy), z potwierdzeniem akcji.
- Mowa: lokalne ASR/TTS.

#### Aktualizacje i ewaluacja

- Dystrybucja modeli (delta updates, wersjonowanie, A/B przez staged rollout), telemetria z poszanowaniem prywatności (federated analytics, differential privacy).
- Metryki: TTFT, tokeny/s, zużycie energii i pamięci, jakość na benchmarkach zadań docelowych po kwantyzacji, testy na flocie urządzeń (low/mid/high-end).

**Źródła:**
- [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [llama.cpp (GitHub)](https://github.com/ggml-org/llama.cpp)

---

<a id="q237"></a>
### 237. Zaprojektuj multimodalny system wyszukiwania (tekst, obraz, wideo)

**Odpowiedź:**

#### Wymagania

Wyszukiwanie po zapytaniu tekstowym lub obrazowym (a także mieszanym) wśród miliardów elementów; niskie opóźnienie (p95 poniżej ~200 ms dla samego wyszukiwania), świeżość indeksu, filtrowanie po metadanych, trafność i bezpieczeństwo treści.

#### Idea: wspólna przestrzeń osadzeń

Model kontrastowy typu **CLIP** uczy się wspólnej przestrzeni wektorowej dla obrazów i tekstu, więc podobieństwo cosinusowe pozwala wyszukiwać obrazy tekstem i odwrotnie:

$$\mathcal{L}=-\frac1N\sum_i\log\frac{\exp(\text{sim}(v_i,t_i)/\tau)}{\sum_j\exp(\text{sim}(v_i,t_j)/\tau)}$$

(symetrycznie po obu stronach).

#### Potok indeksowania

1. **Ingestion**: obrazy, wideo, tekst, audio (transkrypcja ASR), OCR.
2. **Wideo**: podział na sceny/klatki kluczowe (shot detection), embedding klatek i segmentów, agregacja czasowa; transkrypt i opisy generowane przez model VLM jako dodatkowy sygnał.
3. **Embedding** modelami multimodalnymi (CLIP/SigLIP, modele wideo), osobno lub w jednej przestrzeni; metadane (czas, autor, język, uprawnienia).
4. **Indeks ANN** (HNSW, IVF-PQ, FAISS/ScaNN/wektorowe bazy) z kwantyzacją produktową dla oszczędności pamięci; sharding i replikacja.
5. Indeks leksykalny (BM25) dla tekstu, tagów i transkryptów.

#### Potok zapytania

1. Kodowanie zapytania (tekst/obraz) do wektora; opcjonalnie rozszerzanie zapytania (LLM).
2. **Retrieval hybrydowy**: kandydaci z kilku indeksów (dense wizualny, dense tekstowy, BM25) + filtry.
3. **Fuzja** wyników (Reciprocal Rank Fusion).
4. **Reranking** (cross-encoder multimodalny / VLM) na top-$k$ kandydatów.
5. Dla wideo: zwrócenie **znacznika czasu** trafnej sceny.

#### Ewaluacja

- Offline: recall@k, NDCG, MRR na zbiorze zapytań z etykietami; benchmarki typu MSCOCO/Flickr30k retrieval.
- Online: CTR, czas do znalezienia, A/B test, kontrola bezpieczeństwa i uprzedzeń.

#### Skalowanie i pułapki

- Koszt embeddingów wideo (próbkowanie klatek), aktualizacje indeksu (delta + okresowa przebudowa), dryf modelu przy zmianie wersji (reindeksacja).
- „Modality gap": wektory tekstu i obrazu zajmują różne obszary; pomaga fuzja i reranking.
- Prywatność (rozpoznawanie twarzy), prawa autorskie, treści szkodliwe.

**Źródła:**
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734)
- [Sigmoid Loss for Language Image Pre-Training (SigLIP)](https://arxiv.org/abs/2303.15343)
- [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449)

---

<a id="q238"></a>
### 238. Zaprojektuj platformę inferencji LLM (vLLM-as-a-Service)

**Odpowiedź:**

#### Wymagania

Udostępnianie wielu modeli LLM przez API (kompatybilne z OpenAI), streaming, wysoka przepustowość i niski koszt na token, SLO (np. TTFT, TPOT), izolacja klientów, rozliczanie, autoskalowanie.

#### Serce systemu: silnik inferencji (vLLM)

- **PagedAttention**: KV cache dzielony na bloki stronicowane (jak pamięć wirtualna), co niemal eliminuje fragmentację i pozwala na współdzielenie prefiksów (copy-on-write) i większe batche.
- **Continuous batching** (iteration-level scheduling): nowe żądania dołączają do batcha w każdej iteracji, bez czekania na najdłuższą sekwencję.
- **Chunked prefill**, **prefix caching**, **speculative decoding**, kwantyzacja (FP8/INT4), FlashAttention, tensor/pipeline parallelism dla dużych modeli, LoRA multi-adapter serving.

#### Warstwa platformy

```
Klient -> API Gateway (auth, rate limit, billing)
       -> Router / Scheduler (kolejki, priorytety, cache-aware routing)
       -> Pule workerów GPU (vLLM) per model/wersja
       -> Metryki, logi, tracing
```

1. **Gateway**: uwierzytelnianie, kwoty (tokeny/min), walidacja, przypisanie priorytetów (tiery), streaming SSE.
2. **Router**: routing po modelu; **cache-aware / prefix-aware routing** kieruje żądania ze wspólnym prefiksem do tej samej repliki; balansowanie po długości kolejki i wykorzystaniu KV cache.
3. **Rozdzielenie prefill i decode** (disaggregated serving) przy dużej skali i restrykcyjnych SLO.
4. **Autoscaling**: według kolejki, użycia KV cache i opóźnień (nie samego CPU); zimne starty modeli są kosztowne (ładowanie wag rzędu dziesiątek GB), więc warm pools i szybkie ładowanie z lokalnego cache/NVMe.
5. **Zarządzanie modelami**: rejestr, wersjonowanie, canary/A-B rollout, rollback, dystrybucja wag.
6. **Wielodostępność**: limity, izolacja obciążeń, ochrona przed nadużyciami i bardzo długimi promptami.

#### Metryki i SLO

TTFT, TPOT/ITL, throughput (tokeny/s/GPU), wykorzystanie GPU i KV cache, długość kolejki, odsetek błędów/preempcji, koszt na milion tokenów, p50/p95/p99.

#### Kompromisy

- Duży batch zwiększa throughput, ale pogarsza opóźnienie; ustal politykę per tier.
- Kwantyzacja obniża koszt kosztem jakości: waliduj ewaluacjami.
- Preempcja/swap KV cache przy przeciążeniu; admission control i backpressure.
- Niezawodność: health checks, retry z idempotencją, obsługa awarii GPU, degradacja do mniejszego modelu.

**Źródła:**
- [How does vLLM work? (Outcome School)](https://outcomeschool.com/blog/how-does-vllm-work)
- [LLM Inference Optimization (Outcome School)](https://outcomeschool.com/blog/llm-inference-optimization)
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM documentation](https://docs.vllm.ai/)

---

<a id="q239"></a>
### 239. Zaprojektuj platformę do ewaluacji LLM (LLM Evaluation Platform)

**Odpowiedź:**

#### Cel i wymagania

Platforma ma odpowiadać na pytanie „czy nowa wersja modelu / promptu / pipeline'u RAG jest lepsza od poprzedniej i czy nie zepsuła niczego, na czym nam zależy". Wymagania funkcjonalne: zarządzanie zbiorami testowymi, uruchamianie ewaluacji (offline i online), wiele metryk, porównywanie wersji, śledzenie regresji, integracja z CI/CD. Niefunkcjonalne: powtarzalność, koszt (wywołania LLM-judge są drogie), skalowalność (tysiące przypadków × wiele modeli), audytowalność.

#### Architektura

1. **Dataset store** - wersjonowane zbiory (golden set, zbiory adversarialne, przypadki z produkcji po anonimizacji). Każdy przypadek: input, opcjonalny reference, metadane (kategoria, trudność, język).
2. **Runner** - kolejka zadań (np. Kafka/Celery), workery wywołujące system pod testem (model, prompt, agent, RAG) z zapisem pełnych trace'ów (prompt, kontekst, tool calls, latencja, tokeny, koszt). Cache po hashu (model, prompt, parametry) obniża koszt.
3. **Evaluators** (pluginy):
   - metryki deterministyczne: exact match, F1, BLEU/ROUGE, walidacja JSON/schematu, wykonanie kodu i unit testy;
   - metryki dla RAG: faithfulness, answer relevance, context precision/recall (np. RAGAS);
   - **LLM-as-a-judge** - rubryki, pairwise comparison, z losową zamianą kolejności (bias pozycji), kalibracja względem ocen ludzi;
   - human review - kolejka anotacji, wiele oceniających, inter-annotator agreement (Cohen's kappa).
4. **Results store i analityka** - agregacje po segmentach, przedziały ufności (bootstrap), testy istotności, diff między runami, dashboard.
5. **Online evaluation** - logowanie ruchu produkcyjnego, próbkowanie do ewaluacji, A/B testy, feedback użytkowników (thumbs), guardrails i wykrywanie halucynacji.
6. **CI gate** - blokada wdrożenia, gdy metryka spadnie poniżej progu lub wystąpi regresja na krytycznym segmencie.

#### Pułapki

- Judge ma bias (długość odpowiedzi, kolejność, preferowanie własnego modelu) - używaj innego modelu jako sędziego i waliduj na próbce ocenianej przez ludzi.
- Kontaminacja zbioru testowego danymi treningowymi lub promptem.
- Zbyt mały zbiór: różnice w granicach szumu; podawaj CI.
- Niedeterministyczność: temperature 0, kilka próbek, uśrednianie.

#### Weryfikacja platformy

Sprawdź korelację judge'a z ludźmi, stabilność wyników między powtórzeniami oraz czy znane regresje są wykrywane.

**Źródła:**
- [LLM Evaluation (Outcome School)](https://outcomeschool.com/blog/llm-evaluation)
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)
- [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634)

---

<a id="q240"></a>
### 240. Zaprojektuj usługę generowania obrazów z tekstu (text-to-image, w stylu Midjourney)

**Odpowiedź:**

#### Wymagania

Użytkownik podaje prompt tekstowy i po kilkunastu-kilkudziesięciu sekundach dostaje obrazy. Cele: jakość i zgodność z promptem, opóźnienie, przepustowość na GPU, bezpieczeństwo treści, koszt.

#### Model

Standardem są **latent diffusion models**: autoenkoder (VAE) kompresuje obraz do przestrzeni latentnej, a sieć (U-Net lub DiT) uczy się odszumiania w tej przestrzeni, warunkowana embeddingiem tekstu z enkodera (CLIP lub T5). Generowanie: losowy szum -> N kroków odszumiania (sampler, np. DDIM/DPM++) -> dekoder VAE. **Classifier-free guidance** zwiększa zgodność z promptem kosztem różnorodności (`eps = eps_uncond + w*(eps_cond - eps_uncond)`).

#### Architektura systemu

1. **API gateway** - uwierzytelnianie, limity, płatności/kredyty.
2. **Prompt pipeline** - moderacja promptu (klasyfikator + reguły), opcjonalnie rozszerzanie promptu przez LLM, tłumaczenie.
3. **Kolejka zadań** z priorytetami (płatni użytkownicy, tryb szybki/relaks). Zadania są asynchroniczne, klient dostaje status przez WebSocket/polling.
4. **Flota GPU** - workery z załadowanymi wagami; batching żądań o tej samej rozdzielczości, autoskalowanie po długości kolejki. Optymalizacje: fp16/bf16, `torch.compile`, mniej kroków (distillation, np. LCM/turbo), kwantyzacja, cache embeddingów tekstu.
5. **Post-processing** - upscaling (super-resolution), moderacja obrazu (NSFW, przemoc, znaki towarowe), watermark/C2PA.
6. **Storage i CDN** - obrazy w object storage, metadane (prompt, seed, parametry) w bazie.
7. **Feedback loop** - wybory użytkowników (upscale, wariacje, lajki) tworzą dane preferencji do RLHF/DPO dla dyfuzji i do rankingu.

#### Metryki

Offline: FID, CLIP score, ocena preferencji (human eval, pairwise). Online: odsetek wybranych obrazów, retencja, czas do wyniku (p50/p95), koszt GPU na obraz, odsetek odrzuceń moderacji.

#### Ryzyka

Prawa autorskie i dane treningowe, deepfake'i, NSFW, jailbreaki promptów, koszty szczytowe. Warto dodać prywatność (usuwanie promptów) i audyt.

**Źródła:**
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)

---

<a id="q241"></a>
### 241. Zaprojektuj usługę generowania muzyki (w stylu Suno)

**Odpowiedź:**

#### Wymagania

Wejście: prompt opisujący styl/nastrój, opcjonalnie teksty piosenki, referencyjne audio. Wyjście: kilkuminutowy utwór (często z wokalem) w ciągu ~minuty. Kluczowe: jakość muzyczna, spójność struktury (zwrotka/refren), zgodność z promptem, koszt, prawa autorskie.

#### Modelowanie

Dwa główne podejścia:
- **Autoregresyjne modele na tokenach audio**: neuralny kodek (np. EnCodec, SoundStream) kwantyzuje audio do dyskretnych tokenów (RVQ, wiele codebooków), a transformer generuje tokeny warunkowane tekstem (MusicLM, MusicGen). Trudność: długie sekwencje (dziesiątki tokenów na milisekundy razy kilka codebooków) - stosuje się wzorce przeplotu codebooków.
- **Dyfuzja/flow matching w przestrzeni latentnej audio** (spektrogram lub latent kodeka), warunkowana tekstem; lepsza jakość i kontrola, kosztowniejsza w inferencji.

Często pipeline wieloetapowy: (1) LLM generuje tekst/strukturę, (2) model semantyczny planuje strukturę muzyczną, (3) model akustyczny generuje dźwięk, (4) dekoder/vocoder, (5) upsampling i mastering.

#### System

1. Gateway, kredyty, limity.
2. Moderacja tekstu (nienawiść, artyści z nazwiska, chronione teksty).
3. Kolejka asynchroniczna + GPU workery (generacja utworu to zadanie długie - streaming fragmentów pozwala odtwarzać przed końcem).
4. Post-processing: normalizacja głośności, kodowanie do MP3/AAC, watermark audio.
5. Wykrywanie plagiatu/dopasowania do katalogu (audio fingerprinting).
6. Storage, CDN, biblioteka użytkownika, edycja (extend, remix, stems).

#### Ewaluacja

Fréchet Audio Distance, CLAP score (zgodność z tekstem), ocena słuchaczy (MOS, pairwise preference), metryki strukturalne. Online: odsłuchania do końca, zapisy, liczba regeneracji.

#### Ryzyka

Licencje danych treningowych, klonowanie głosu, koszt inferencji, nadużycia. Rozwiązania: dane licencjonowane, watermarking, filtry.

**Źródła:**
- [MusicLM: Generating Music From Text](https://arxiv.org/abs/2301.11325)
- [Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284)
- [High Fidelity Neural Audio Compression (EnCodec)](https://arxiv.org/abs/2210.13438)

---

<a id="q242"></a>
### 242. Zaprojektuj usługę generowania wideo (w stylu Sora)

**Odpowiedź:**

#### Wymagania

Text-to-video (i image-to-video) o długości kilku-kilkudziesięciu sekund, rozdzielczości HD, spójny w czasie i fizycznie wiarygodny. Trudności: ogromna liczba danych na klip, koszt GPU rzędu minut na krótki film, spójność obiektów między klatkami.

#### Model

- **Kompresja**: wideo trafia do enkodera (video VAE), który redukuje je przestrzennie i czasowo do latentów; następnie dzieli się je na **spacetime patches** (tokeny).
- **Generator**: Diffusion Transformer (DiT) odszumiający patche, warunkowany tekstem (embeddingi z T5/CLIP lub opisy generowane przez LLM). Trening na wielu rozdzielczościach i długościach.
- **Dane**: gęste, automatycznie generowane podpisy (recaptioning) poprawiają zgodność z promptem.
- Kaskada: generacja krótkiego klipu niskiej rozdzielczości -> super-resolution przestrzenna -> interpolacja klatek.

#### System

1. API i kolejka zadań (generacja trwa minuty - tylko asynchronicznie, webhooki/powiadomienia).
2. Moderacja promptu, obrazów wejściowych (osoby publiczne, treści szkodliwe) i wyniku.
3. Klaster GPU/TPU z model parallelism (sequence/tensor parallel), bo pamięć na tokeny wideo jest ogromna; harmonogram zadań według priorytetu, przewidywanego czasu i kosztu.
4. Optymalizacje: mniej kroków samplera (distillation), kwantyzacja, cache embeddingów, klasyfikatory jakości do wczesnego odrzucania złych sampli.
5. Post-processing: upscaling, kodowanie (H.264/H.265), dźwięk (osobny model), watermark i metadane C2PA.
6. Storage + CDN, wersjonowanie, edycja (extend, remix, storyboard).

#### Ewaluacja

FVD, CLIP-based alignment, benchmarki spójności (VBench), ocena ludzi (pairwise). Online: odsetek regeneracji, retencja, koszt na sekundę wideo.

#### Ryzyka

Deepfake'i i dezinformacja (watermark, provenance, ograniczenia na realistyczne osoby), prawa autorskie, koszty i emisje, limity użycia.

**Źródła:**
- [Video generation models as world simulators (OpenAI)](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Video Diffusion Models](https://arxiv.org/abs/2204.03458)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)

---

<a id="q243"></a>
### 243. Zaprojektuj agenta programistycznego AI (AI Coding Agent)

**Odpowiedź:**

#### Idea

Agent programistyczny to LLM działający w pętli: **obserwuj -> zaplanuj -> użyj narzędzia -> zobacz wynik -> powtórz**, aż zadanie zostanie ukończone. Przykłady: Claude Code, Cursor.

#### Komponenty

1. **Model** (LLM z tool use) - wybór modelu zależnie od kroku (silny do planowania, tańszy do prostych edycji).
2. **Narzędzia**: odczyt/zapis/edycja plików (edycja przez diff/search-replace, nie przepisywanie całych plików), wyszukiwanie (grep, glob, semantyczne), shell (testy, linter, build), git, przeglądarka/dokumentacja, MCP dla zewnętrznych integracji.
3. **Kontekst**: okno jest ograniczone, więc potrzebne są: indeks kodu (chunking po AST, embeddingi + BM25, wyszukiwanie hybrydowe), mapa repozytorium, pliki instrukcji projektu (np. CLAUDE.md), kompaktowanie/streszczanie historii, pamięć długoterminowa.
4. **Planowanie i pętla**: lista zadań, podział na subagenty (eksploracja, implementacja, review) z osobnym kontekstem; ograniczenie liczby kroków i budżetu.
5. **Sandbox i bezpieczeństwo**: uruchamianie w kontenerze, system uprawnień (zatwierdzanie zapisów i komend), allowlisty, ochrona przed prompt injection z treści repo/stron, sekrety poza kontekstem.
6. **Weryfikacja**: uruchamianie testów, typecheck i lintera po edycji; pętla naprawcza na podstawie błędów - to największy czynnik jakości.
7. **UX**: streaming, podgląd diffów, cofanie zmian (checkpointy/git), integracja z IDE.

#### Ewaluacja

SWE-bench (rozwiązywanie realnych issue), własne zestawy zadań z repozytoriami firmy, metryki: odsetek zadań z przechodzącymi testami, liczba kroków i tokenów, czas, odsetek zaakceptowanych zmian, regresje. Trace każdego przebiegu jest przechowywany do analizy błędów.

#### Pułapki

Halucynowane API, pętle bez postępu, zbyt duże zmiany, wyciek sekretów, koszt tokenów. Zaczynaj od prostych, dobrze narzędziowanych pętli, a złożoność dodawaj tylko gdy poprawia wyniki.

**Źródła:**
- [How does Claude Code work? (Outcome School)](https://outcomeschool.com/blog/how-does-claude-code-work)
- [How does Cursor work? (Outcome School)](https://outcomeschool.com/blog/how-does-cursor-work)
- [Building effective agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://arxiv.org/abs/2310.06770)

---

<a id="q244"></a>
### 244. Zaprojektuj system ML do rekomendacji filmów na YouTube

**Odpowiedź:**

#### Cel i metryki

Cel biznesowy: zwiększyć satysfakcję i czas oglądania, a nie same kliknięcia (clickbait). Metryki online: watch time, sesje, ankiety satysfakcji, retencja, odsetek „nie interesuje mnie". Offline: recall@k, NDCG, AUC, logloss dla poszczególnych głów (klik, obejrzenie >X%, lajk).

#### Architektura wieloetapowa

Miliardy filmów uniemożliwiają skorowanie wszystkiego, dlatego stosuje się lejek:

1. **Candidate generation (retrieval)** - kilkaset kandydatów z miliardów, w kilkadziesiąt ms. Model **two-tower**: wieża użytkownika (historia oglądania, wyszukiwania, demografia, kontekst) i wieża filmu (embedding ID, tytuł, kanał, kategoria). Trening z softmax na negatywach in-batch (z korektą popularności) lub sampled softmax. Indeks ANN (ScaNN/FAISS/HNSW) dla embeddingów filmów. Dodatkowe źródła: subskrypcje, popularne lokalnie, filmy podobne do ostatnio oglądanych.
2. **Ranking** - ciężki model (głęboka sieć, np. DCN/Wide&Deep) na setkach kandydatów; cechy: użytkownik, film, kontekst, interakcje. Wielozadaniowość (multi-task, MMoE): P(klik), oczekiwany czas oglądania, P(lajk), P(dislike); wynik to ważona kombinacja.
3. **Re-ranking** - różnorodność (MMR), świeżość, reguły polityki, ograniczenia treści szkodliwych, kanały, brak duplikatów.

#### Dane i cechy

Logi impresji i zachowań (pozytywy: obejrzenie znacznej części; negatywy: impresje bez interakcji - uwaga na bias pozycji). Cechy świeżości (age filmu jest ważny), cechy sekwencyjne (Transformer/GRU na historii). Embeddingi treści (wideo, dźwięk, tekst) pomagają w cold-start.

#### Trening i serwowanie

Ciągły retraining (codziennie/częściej), feature store, spójność cech offline/online (training-serving skew), serwowanie z latencją < 200 ms end-to-end.

#### Wyzwania

Cold start (nowy film/użytkownik), feedback loop i bąble filtrujące, bias pozycji (position as a feature/IPW), eksploracja (bandity), dobrostan użytkownika. Weryfikacja: A/B testy z metrykami długoterminowymi i holdoutem.

**Źródła:**
- [Deep Neural Networks for YouTube Recommendations (Google Research)](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)
- [Machine Learning Systems Design (Chip Huyen)](https://huyenchip.com/machine-learning-systems-design/toc.html)

---

<a id="q245"></a>
### 245. Zaprojektuj system ML do wyszukiwania filmów na YouTube

**Odpowiedź:**

#### Problem

Dla zapytania tekstowego zwrócić uporządkowaną listę filmów. Różnica względem rekomendacji: intencja jest jawna (zapytanie), więc kluczowa jest trafność zapytanie-dokument, a personalizacja to dodatek.

#### Rozumienie zapytania

Normalizacja, korekta literówek, tłumaczenie/wykrywanie języka, klasyfikacja intencji (nawigacyjna, informacyjna, rozrywkowa), rozszerzanie o synonimy, autouzupełnianie.

#### Retrieval

- **Leksykalny**: indeks odwrócony (BM25) na tytule, opisie, transkrypcji (ASR), tagach.
- **Semantyczny**: dwuwieżowy embedding zapytania i filmu (tekst + cechy wizualne/audio) z ANN (HNSW/ScaNN). Trening kontrastowy na parach (zapytanie, film kliknięty i obejrzany).
- Łączenie wyników (hybrid, RRF).

#### Ranking

Learning to rank: model (LambdaMART lub sieć neuronowa) z cechami: dopasowanie tekstowe pól, embeddingowe podobieństwo, jakość i popularność filmu, świeżość, współczynnik klikalności i czas oglądania po zapytaniu, sygnały autorytetu kanału, personalizacja (język, historia). Etykiety: niejawne z logów (klik, watch time) z korektą bias pozycji, uzupełnione ocenami ludzi (relevance raters) dla jakości.

Kolejne etapy: cross-encoder do re-rankingu top-N, filtry bezpieczeństwa i polityk, różnorodność.

#### Ewaluacja

Offline: NDCG@k, MRR na zbiorze ocenionym przez ludzi; online: CTR, watch time, odsetek zapytań bez kliknięcia, reformulacje zapytań (sygnał niezadowolenia).

#### Wyzwania

Latencja (dziesiątki ms na retrieval), świeżość indeksu (nowe filmy), long tail zapytań, wielojęzyczność, spam i clickbait, cold start filmów bez sygnałów.

**Źródła:**
- [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632)
- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
- [How does a Reranker work? (Outcome School)](https://outcomeschool.com/blog/how-does-a-reranker-work)
- [Learning to rank (Wikipedia)](https://en.wikipedia.org/wiki/Learning_to_rank)

---

<a id="q246"></a>
### 246. Zaprojektuj system ML do spersonalizowanego feedu treści

**Odpowiedź:**

#### Cel

Ułożyć dla każdego użytkownika kolejność postów (znajomi, strony, reklamy, filmy), maksymalizując długoterminową wartość: zaangażowanie znaczące (komentarze, udostępnienia), retencję i dobrostan, a nie surowy czas.

#### Pipeline

1. **Inventory / candidate generation** - posty od znajomych i obserwowanych (świeże, ograniczone czasowo), rekomendacje spoza sieci (two-tower + ANN), trendy. Kilka tysięcy kandydatów.
2. **Wstępne rankowanie** - lekki model, redukcja do setek.
3. **Główny ranking** - wielozadaniowa sieć przewiduje wiele zdarzeń: klik, lajk, komentarz, udostępnienie, ukrycie, zgłoszenie, dłuższe zatrzymanie. Wynik = ważona suma (wagi dobrane pod cele produktu i kalibrowane A/B) z ewentualnymi karami za negatywne sygnały.
4. **Re-ranking** - różnorodność autorów i formatów, ograniczenia częstotliwości, integralność (obniżanie clickbaitu, dezinformacji), wstrzykiwanie reklam.

#### Cechy

Użytkownik (historia interakcji, zainteresowania), autor (relacja, częstość kontaktu), post (typ, treść - embeddingi tekstu i obrazu, świeżość, popularność), kontekst (urządzenie, pora dnia). Cechy sekwencyjne z Transformerem. Cechy odziedziczone z grafu społecznego (GNN).

#### Trening

Logi impresji z etykietami wielozadaniowymi; korekta bias pozycji; negatywy = impresje bez interakcji. Retraining ciągły/dzienny; feature store i spójność online/offline.

#### Metryki

Offline: AUC/logloss/kalibracja per zadanie, NDCG. Online: A/B - dzienne aktywne osoby, znaczące interakcje, czas, retencja 7/30 dni, skargi/ukrycia, metryki dobrostanu. Ważna kalibracja prawdopodobieństw, bo wynik łączy wiele głów.

#### Wyzwania

Bąble filtrujące, feedback loops, cold start, sprawiedliwość wobec twórców, latencja (kilkaset ms), koszt serwowania dużych modeli. Rozwiązania: eksploracja (bandit), quota różnorodności, metryki długoterminowe, holdouty.

**Źródła:**
- [Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM)](https://arxiv.org/abs/1906.00091)
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)

---

<a id="q247"></a>
### 247. Zaprojektuj system ML do wykrywania szkodliwych treści (harmful content detection)

**Odpowiedź:**

#### Problem

Automatyczne wykrywanie treści naruszających zasady: mowa nienawiści, przemoc, nagość, samookaleczenia, spam, dezinformacja, w tekstach, obrazach, wideo i audio, na dużą skalę i często w czasie rzeczywistym.

#### Specyfika

- Bardzo **niezbalansowane klasy** (szkodliwe treści to ułamek procenta).
- Koszty błędów asymetryczne i zależne od kategorii: dla CSAM/terroryzmu priorytetem jest recall, dla mniej groźnych - precyzja (fałszywe usunięcia niszczą zaufanie).
- **Adversarial**: autorzy zmieniają pisownię, obrazy, kodują treści.
- Definicje zależne od polityki, języka i kontekstu kulturowego.

#### Architektura

1. **Wejście**: post w chwili publikacji (synchronicznie dla najgroźniejszych klas), zgłoszenia użytkowników, ponowne skanowanie.
2. **Modele per modalność**: tekst (fine-tuned multilingual transformer, np. XLM-R), obraz (CNN/ViT), wideo (próbkowanie klatek + audio/ASR), multimodalne (fuzja - memy, gdzie sens zależy od obrazu i podpisu razem). Sygnały uzupełniające: reputacja autora, zachowanie konta, graf.
3. **Hashing i dopasowanie** znanych szkodliwych treści (perceptual hashing) - szybkie i tanie.
4. **Wyniki** -> progi per kategoria: auto-usunięcie (wysoka pewność), do kolejki moderatorów (środek), pozostawienie.
5. **Human review**: priorytetyzacja kolejki po severity i zasięgu, odwołania; decyzje wracają jako etykiety (active learning).
6. **Feedback**: retraining, wykrywanie driftu, red teaming.

#### Ewaluacja

Precision/recall per kategoria i język, PR-AUC (nie accuracy), prevalence w próbce losowej (metryka najważniejsza: ile szkodliwych treści zostaje widzianych), czas do usunięcia, odsetek uchylonych odwołań, analiza fairness (błędy dla dialektów i grup).

#### Wyzwania

Mało etykiet (weak supervision, augmentacja, LLM do wstępnego etykietowania), zdrowie psychiczne moderatorów, przejrzystość, zgodność z regulacjami (DSA).

**Źródła:**
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)](https://arxiv.org/abs/1911.02116)
- [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406)
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)

---

<a id="q248"></a>
### 248. Zaprojektuj system ML do rekomendacji podobnych ofert (Similar Listings) na Airbnb

**Odpowiedź:**

#### Problem

Na stronie oferty pokazujemy „podobne oferty", które zwiększają szansę na rezerwację (np. gdy termin niedostępny albo oferta nie spełnia oczekiwań). To rekomendacja typu item-to-item, zależna od kontekstu sesji.

#### Podejście: embeddingi ofert

Zainspirowane word2vec (skip-gram): sesje użytkowników (sekwencje klikniętych ofert) traktujemy jak „zdania", a oferty jak „słowa". Model uczy się wektorów tak, by oferty pojawiające się w podobnych kontekstach były blisko. Modyfikacje istotne dla domeny:
- **Rezerwacja jako globalny kontekst** - oferta zarezerwowana na końcu sesji jest zawsze pozytywnym sąsiadem.
- **Negatywy z tego samego rynku** (miasto), żeby embeddingi rozróżniały oferty wewnątrz lokalizacji, nie tylko między miastami.
- **Cold start**: nowa oferta bez kliknięć dostaje wektor uśredniony z podobnych ofert (cena, typ, lokalizacja, liczba sypialni).
- Krótkie sesje i rzadkie oferty: filtrowanie, minimalna liczba wystąpień.

#### Serwowanie

1. **Retrieval**: kandydaci = ANN (HNSW/FAISS) po embeddingu, z filtrami twardymi: ta sama lokalizacja/rynek, dostępność w wybranych datach, liczba gości, przedział cenowy.
2. **Ranking**: model gradient boosting lub sieć na cechach (podobieństwo embeddingów, różnica ceny, ocena, liczba zdjęć, odległość, jakość gospodarza, historia użytkownika). Etykieta: klik lub rezerwacja w sesji po pokazaniu.
3. **Re-ranking**: różnorodność (nie same identyczne apartamenty), reguły biznesowe.

#### Metryki

Offline: recall@k na sesjach z rezerwacją (czy zarezerwowana oferta jest w top-k), NDCG. Online: CTR na sekcji, współczynnik rezerwacji przypisany sekcji, przychód, A/B.

#### Wyzwania

Dwustronny marketplace (sprawiedliwość dla gospodarzy), sezonowość i dostępność w czasie, uczenie embeddingów przy rzadkości rezerwacji, aktualizacja dzienna i szybkie ANN.

**Źródła:**
- [Applying Deep Learning To Airbnb Search](https://arxiv.org/abs/1810.09591)
- [Efficient Estimation of Word Representations in Vector Space (word2vec)](https://arxiv.org/abs/1301.3781)
- [Efficient and robust approximate nearest neighbor search using HNSW](https://arxiv.org/abs/1603.09320)

---

<a id="q249"></a>
### 249. Zaprojektuj system ML do rekomendacji produktów zastępczych (Replacement Product Recommendation)

**Odpowiedź:**

#### Problem

Gdy produkt jest niedostępny (np. w zamówieniu zakupów spożywczych), system proponuje zamiennik. Ma być akceptowalny dla klienta: podobna funkcja, zbliżona cena i marka, te same ograniczenia (dieta, alergeny, rozmiar).

#### Definicja podobieństwa

- **Substytuty** (ten sam cel, wymienne) różnią się od **komplementów** (kupowane razem). Modele oparte tylko na współkupowaniu mylą je, więc potrzebne są sygnały substytucji: historia „produkt A został zastąpiony przez B i klient zaakceptował", kliknięcia w alternatywy.
- Twarde reguły: alergeny, wegetariańskie/halal, rozmiar/jednostka, kategoria.

#### Architektura

1. **Kandydaci**: z tej samej kategorii i sklepu/magazynu w stanie dostępnym; ANN po embeddingach produktu (tekst nazwy/opisu przez transformer, obraz, atrybuty, cena) uczonych kontrastowo na parach zaakceptowanych zamienników.
2. **Ranking**: model (GBDT lub sieć) z cechami: podobieństwo embeddingów, różnica ceny (względna), ta sama marka, zgodność atrybutów (gramatura, smak), popularność zamiennika, historia użytkownika (preferencje marek, akceptowane zamienniki), dostępność. Etykieta: klient zaakceptował zamiennik (lub nie zwrócił go).
3. **Personalizacja**: wagi cech zależą od klienta (wrażliwy na cenę vs. lojalny wobec marki).
4. **Walidacja i reguły biznesowe**: marża, dostępność, kary za produkt o gorszej jakości.

#### Metryki

Offline: recall@k/precision@k względem zamienników zaakceptowanych przez klientów, NDCG. Online: odsetek akceptacji, skargi, zwroty, ocena zamówienia, przychód, satysfakcja (ankieta); A/B.

#### Wyzwania

Cold start nowych produktów (embeddingi z treści), etykiety obciążone tym, co system dotąd proponował (bias selekcji; eksploracja losowa lub IPS), silna lokalność zapasów, zmienna dostępność - szybka aktualizacja indeksu.

**Źródła:**
- [Learning to rank (Wikipedia)](https://en.wikipedia.org/wiki/Learning_to_rank)
- [Efficient and robust approximate nearest neighbor search using HNSW](https://arxiv.org/abs/1603.09320)
- [A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)](https://arxiv.org/abs/2002.05709)
- [Recommender system (Wikipedia)](https://en.wikipedia.org/wiki/Recommender_system)

---

<a id="q250"></a>
### 250. Zaprojektuj system ML do rekomendacji wydarzeń (Event Recommendation)

**Odpowiedź:**

#### Specyfika

Wydarzenia (koncerty, meetupy) mają krótkie życie, są związane z lokalizacją i czasem, mają ograniczoną pojemność i **nie mają historii interakcji**, gdy się pojawiają. Klasyczna współfiltracja zawodzi, więc dominuje podejście content-based + kontekst + sygnały społeczne.

#### Kandydaci

- Filtry twarde: lokalizacja (promień od użytkownika lub od miejsca, które odwiedza), data w przyszłości, dostępność miejsc, ograniczenia wiekowe.
- Źródła: kategorie zainteresowań, wydarzenia znajomych i organizatorów obserwowanych, popularne w okolicy, podobne do przeszłych uczestnictw.

#### Ranking

Model wielozadaniowy przewiduje P(zainteresowanie/„interested"), P(rejestracja), P(uczestnictwo). Cechy:
- **użytkownik**: historia uczestnictwa, kategorie, lokalizacje, godziny aktywności,
- **wydarzenie**: embedding treści (tytuł, opis, kategoria), organizator, cena, pojemność, czas do wydarzenia, liczba zapisanych, tempo przyrostu zapisów,
- **relacje**: odległość, czy uczestniczą znajomi (graf społeczny), historia z organizatorem,
- **kontekst**: dzień tygodnia, sezon, pogoda.

Cold start rozwiązują embeddingi treści i cechy organizatora; eksploracja przez bandity, by wydarzenia bez sygnałów zdobyły dane.

#### Re-ranking

Różnorodność kategorii, unikanie kolizji czasowych, pilność (wydarzenie za 2 dni), limity częstotliwości powiadomień; osobny kanał powiadomień push z optymalizacją czasu wysyłki.

#### Metryki

Offline: NDCG, AUC per zdarzenie (z podziałem czasowym, nie losowym, bo przyszłość ma być nieznana). Online: rejestracje, frekwencja rzeczywista, CTR, odsetek wyłączonych powiadomień, przychód z biletów; A/B.

#### Wyzwania

Krótki cykl życia, rzadkość danych, zależność od lokalizacji, sprawiedliwość wobec małych organizatorów, prognozowanie popytu do pojemności.

**Źródła:**
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- [Recommender system (Wikipedia)](https://en.wikipedia.org/wiki/Recommender_system)
- [Machine Learning Systems Design (Chip Huyen)](https://huyenchip.com/machine-learning-systems-design/toc.html)

---

<a id="q251"></a>
### 251. Zaprojektuj system ML do wyszukiwania multimodalnego (Multimodal Search)

**Odpowiedź:**

#### Problem

Wyszukiwanie, w którym zapytanie i/lub dokumenty mogą być tekstem, obrazem, wideo lub audio: „znajdź zdjęcia plaż o zachodzie słońca", „znajdź produkt podobny do tego zdjęcia + w kolorze niebieskim".

#### Rdzeń: wspólna przestrzeń embeddingów

Model dwuwieżowy trenowany kontrastowo (CLIP-style): enkoder tekstu i enkoder obrazu (ViT) mapują do jednej przestrzeni; strata InfoNCE zbliża pasujące pary i oddala niepasujące w batchu:

`L = -log( exp(sim(q,d+)/tau) / sum_j exp(sim(q,d_j)/tau) )`

Dla wideo: embeddingi próbkowanych klatek + audio/ASR agregowane w wektor. Dane: pary obraz-podpis, kliknięcia użytkowników z logów wyszukiwania.

#### Architektura

1. **Indeksowanie offline**: dla każdego zasobu embedding (wraz z metadanymi i tekstem OCR/opisem) -> indeks ANN (HNSW/IVF-PQ, FAISS/ScaNN) + klasyczny indeks tekstowy.
2. **Zapytanie online**: enkoder odpowiedniej modalności (lub kombinacja - fuzja embeddingów tekstu i obrazu dla zapytań mieszanych), retrieval top-K przez ANN, plus wyniki leksykalne (hybrid).
3. **Re-ranking**: cross-modalny model (cross-encoder/VLM) na top-N, cechy jakości i popularności, personalizacja, filtry (bezpieczeństwo, licencje).
4. **Serwowanie**: kwantyzacja i sharding indeksu, cache popularnych zapytań, latencja rzędu dziesiątek ms na ANN.

#### Ewaluacja

Recall@K, MRR, NDCG na zbiorach z ocenami relewancji (zapytania z logów), analiza per modalność i typ zapytania; online: CTR, sukces sesji, odsetek reformulacji.

#### Wyzwania

Modality gap (embeddingi różnych modalności mimo treningu zajmują osobne obszary), wielojęzyczność, koszt indeksowania dużych zbiorów wideo, aktualizacje indeksu, uprzedzenia i bezpieczeństwo treści.

**Źródła:**
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734)
- [Accelerating Large-Scale Inference with Anisotropic Vector Quantization (ScaNN)](https://arxiv.org/abs/1908.10396)
- [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406)

---

<a id="q252"></a>
### 252. Zaprojektuj system ML do przewidywania kliknięć w reklamy (Ad Click Prediction)

**Odpowiedź:**

#### Problem

Dla zapytania/kontekstu i kandydatów reklamowych przewidzieć P(klik) (pCTR), a często też P(konwersja). Wynik trafia do aukcji: ranking po `eCPM = pCTR * bid * 1000`. Model musi być dobrze **skalibrowany**, bo prawdopodobieństwa wchodzą do rozliczeń, nie tylko do rankingu.

#### Dane i cechy

- Ekstremalnie rzadkie, wysokowymiarowe cechy kategoryczne: ID reklamy, reklamodawcy, kreacji, strony, użytkownika, słowa zapytania. Stosuje się embeddingi i hashing trick.
- Cechy liczbowe: historyczne CTR (z wygładzaniem), pozycja, pora dnia, urządzenie.
- Cechy interakcji (cross features): użytkownik x kategoria reklamy.
- Silnie niezbalansowane etykiety (CTR zwykle ułamek procenta): downsampling negatywów wymaga późniejszej korekty kalibracji (`p = p'/(p' + (1-p')/w)`).

#### Modele

Ewolucja: regresja logistyczna z cechami krzyżowymi i FTRL-Proximal (online, rzadkie wagi) -> Factorization Machines -> Wide & Deep, DeepFM, DCN, DLRM (embeddingi + interakcje cech + MLP). Strata: logloss. Ważny jest **position bias** (reklamy wyżej klikane niezależnie od jakości): pozycja jako cecha przy treningu i ustalona/marginalizowana w serwowaniu.

#### Trening i serwowanie

Dane strumieniowe, aktualizacje online lub częste retrainingi (godziny), opóźnione konwersje (delayed feedback), wersjonowanie i canary. Latencja: kilka-kilkanaście ms na reklamę, kandydaci wstępnie filtrowani. Ogromne tablice embeddingów wymagają sharding/parameter server.

#### Metryki

Offline: AUC, logloss, **calibration** (stosunek średniej predykcji do CTR), normalized entropy. Online: CTR, przychód, RPM, koszt dla reklamodawców, A/B na małym ułamku ruchu.

#### Wyzwania

Cold start reklam, eksploracja (bandit/Thompson sampling), zmiany dystrybucji (sezonowość, kampanie), prywatność (ograniczone identyfikatory), fraud kliknięć.

**Źródła:**
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)
- [Deep Learning Recommendation Model for Personalization and Recommendation Systems (DLRM)](https://arxiv.org/abs/1906.00091)
- [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)

---

<a id="q253"></a>
### 253. Zaprojektuj system ML do szacowania czasu dostawy (Estimate Delivery Time)

**Odpowiedź:**

#### Problem

Przewidzieć czas od złożenia zamówienia do dostarczenia (np. jedzenie, paczka). To zadanie **regresji** (często prognoza rozkładu, nie pojedynczej liczby), a od jakości zależą zaufanie klientów, przydział kurierów i cena.

#### Składowe czasu

Czas = przygotowanie zamówienia w restauracji + oczekiwanie na kuriera + dojazd do restauracji + odbiór + dojazd do klienta + wydanie. Można modelować składowe osobno (interpretowalność) lub całość end-to-end.

#### Cechy

- **Zamówienie**: liczba i rodzaj pozycji, wartość, złożoność.
- **Sklep/restauracja**: historyczny czas przygotowania (średnia, percentyle), obciążenie w tej chwili (liczba zamówień w kolejce), godziny szczytu.
- **Geografia**: odległość (drogowa, nie prosta), czas przejazdu z API map/ruch, typ zabudowy, piętro.
- **Podaż kurierów**: liczba dostępnych, ich odległość, średnia prędkość.
- **Kontekst**: pora dnia, dzień tygodnia, pogoda, święta, wydarzenia.
- **Cechy czasu rzeczywistego** ze streamingu (Kafka/Flink) i feature store.

#### Model

Gradient boosting (XGBoost/LightGBM) jest silnym punktem odniesienia dla danych tabelarycznych; sieci neuronowe (embeddingi lokalizacji, sekwencje) dodają zysk przy dużej skali. Strata: MAE/Huber; ze względu na doświadczenie klienta często **asymetryczna** (spóźnienie gorsze niż wcześniejsza dostawa) lub regresja kwantylowa, by pokazywać przedział/„do 35 min" (np. kwantyl 0,8-0,9). Estymata jest aktualizowana w trakcie realizacji.

#### Metryki

MAE/RMSE, MAPE, pokrycie przedziałów (czy P90 rzeczywiście obejmuje ~90%), odsetek spóźnień powyżej obiecanego czasu, oddzielnie w szczycie i w segmentach. Online: NPS, anulowania, skargi, zaufanie do ETA.

#### Wyzwania

Cenzurowanie i ogony rozkładu, shift sezonowy i pogodowy, sprzężenie zwrotne (obiecany czas wpływa na zachowanie kurierów), spójność cech online/offline, monitorowanie driftu i szybki fallback do prostszego modelu.

**Źródła:**
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [scikit-learn: Gradient boosting (quantile regression)](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting)
- [Quantile regression (Wikipedia)](https://en.wikipedia.org/wiki/Quantile_regression)

---

<a id="q254"></a>
### 254. Zaprojektuj system ML do wyszukiwania obrazów (Image Search)

**Odpowiedź:**

#### Warianty

1. **Text-to-image**: zapytanie tekstowe -> obrazy.
2. **Image-to-image (visual search)**: obraz jako zapytanie -> podobne wizualnie (np. produkty).
Oba oparte na embeddingach i wyszukiwaniu najbliższych sąsiadów.

#### Modele embeddingowe

- Dla podobieństwa wizualnego: CNN/ViT trenowany metric learning (triplet loss, contrastive, ArcFace-like) na parach/trójkach (kotwica, pozytyw, negatyw); pozytywy z augmentacji, tych samych produktów, kliknięć.
- Dla wyszukiwania tekstowego: dwuwieżowy model kontrastowy (CLIP) - wspólna przestrzeń tekst-obraz.

#### Architektura

1. **Offline (indeksowanie)**: pipeline przetwarza nowe obrazy: dekodowanie, deduplikacja (perceptual hash), OCR/tagowanie/detekcja obiektów, embedding (batch na GPU) -> zapis w indeksie wektorowym (HNSW lub IVF-PQ w FAISS/ScaNN; sharding, kwantyzacja produktowa dla pamięci) + metadane w bazie.
2. **Online**: zapytanie -> embedding -> ANN top-K (kilka ms-dziesiątki ms) -> filtrowanie po metadanych (licencje, bezpieczeństwo) -> re-ranking.
3. **Re-ranking**: model uczący się rankingu z cechami: podobieństwo, jakość obrazu (estetyka, rozdzielczość), popularność, CTR, sygnały tekstowe strony, personalizacja. Etykiety z kliknięć (z korektą bias pozycji) i od raterów.
4. **Pętla danych**: logi zapytań i kliknięć -> fine-tuning embeddingów, hard negatives mining.

#### Metryki

Offline: recall@K, mAP, NDCG na zbiorze zapytań z ocenami; dla ANN dodatkowo recall względem dokładnego kNN (kompromis dokładność vs. latencja). Online: CTR, czas do pierwszego kliknięcia, odsetek zapytań bez interakcji.

#### Wyzwania

Skala (miliardy obrazów), aktualizacje indeksu na bieżąco, treści szkodliwe i prawa autorskie, uprzedzenia w wynikach, wielojęzyczność, zapytania niejednoznaczne (różnorodność wyników).

**Źródła:**
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)
- [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734)
- [FaceNet: A Unified Embedding for Face Recognition and Clustering (triplet loss)](https://arxiv.org/abs/1503.03832)
- [Efficient and robust approximate nearest neighbor search using HNSW](https://arxiv.org/abs/1603.09320)

---

<a id="q255"></a>
### 255. Zaprojektuj system ML do rekomendacji znajomych (Friends Recommendation)

**Odpowiedź:**

#### Problem

Zaproponować użytkownikowi osoby, które może znać („People You May Know"). To **link prediction** w grafie społecznym: przewidzieć krawędź (znajomość), której jeszcze nie ma.

#### Generowanie kandydatów

- **Znajomi znajomych** (2-hop): najsilniejszy sygnał, ale ogromne fan-outy przy węzłach o wysokim stopniu - próbkowanie i limity.
- Wspólne organizacje: szkoła, praca, grupy, wydarzenia; kontakty z książki adresowej (za zgodą).
- Embeddingi grafowe (node2vec, GraphSAGE/GNN) + ANN - znajdują podobnych, nawet bez wspólnych znajomych.
- Osoby, które odwiedziły profil użytkownika, lub użytkownik ich profil.

#### Ranking

Model klasyfikacji par (u, v), przewiduje P(wyślij/zaakceptuj zaproszenie). Cechy:
- grafowe: liczba wspólnych znajomych, Jaccard, Adamic-Adar, embeddingi GNN,
- profilowe: wspólna lokalizacja, szkoła, praca, wiek, język,
- behawioralne: interakcje, wizyty profilu, aktywność, „recenzja wzajemna" (czy v niedawno oglądał u),
- kontekstowe: świeżość konta, liczba już wysłanych zaproszeń.
Modele: GBDT lub sieć; etykieta: zaproszenie wysłane i **zaakceptowane** (samo wysłanie może być spamem). Trening z negatywami (losowe + trudne: 2-hop bez zaproszenia).

#### Wymagania szczególne

- **Wzajemność i dwustronność**: optymalizuj P(akceptacji) i satysfakcję odbiorcy, nie tylko nadawcy; limity, by nie zasypywać popularnych profili prośbami.
- **Prywatność i bezpieczeństwo**: nie ujawniać, kto odwiedzał czyj profil bez zgody; unikanie sugerowania osób blokowanych, ofiar prześladowania; wykrywanie spamerów i fałszywych kont.
- Cold start: nowi użytkownicy bez grafu - kontakty, profil, import.

#### Metryki

Offline: AUC/PR, recall@k dla przyszłych znajomości (podział czasowy). Online: liczba zaakceptowanych zaproszeń, odsetek odrzuceń/blokad/zgłoszeń, wzrost gęstości grafu i retencji; A/B.

#### Wyzwania

Skala grafu (miliardy węzłów), aktualizacje w czasie zbliżonym do rzeczywistego, bias popularności, wpływ na wzrost sieci i różnorodność.

**Źródła:**
- [Inductive Representation Learning on Large Graphs (GraphSAGE)](https://arxiv.org/abs/1706.02216)
- [node2vec: Scalable Feature Learning for Networks](https://arxiv.org/abs/1607.00653)
- [Graph Convolutional Neural Networks for Web-Scale Recommender Systems (PinSage)](https://arxiv.org/abs/1806.01973)
- [Link prediction (Wikipedia)](https://en.wikipedia.org/wiki/Link_prediction)

---

<a id="q256"></a>
### 256. Zaprojektuj system rekomendacji produktów dla platformy e-commerce

**Odpowiedź:**

#### Cele i przypadki użycia

Różne powierzchnie mają różne cele: strona główna („dla Ciebie"), strona produktu („podobne", „kupowane razem"), koszyk (cross-sell), e-mail/push. Metryki: konwersja, AOV, przychód/marża, retencja, różnorodność.

#### Dane i cechy

Zdarzenia (wyświetlenie, klik, dodanie do koszyka, zakup, zwrot), katalog (tytuł, opis, obrazy, kategoria, cena), użytkownik (historia, segment), kontekst (urządzenie, sezon). Sygnały o różnej sile: zakup > koszyk > klik > wyświetlenie.

#### Architektura

1. **Kandydaci (retrieval)** - kilka źródeł łączonych:
   - współfiltracja: macierzowa faktoryzacja/ALS na macierzy użytkownik-produkt (dane niejawne),
   - two-tower + ANN (embeddingi użytkownika z sekwencji zdarzeń, np. Transformer/GRU4Rec; embeddingi produktów),
   - item-to-item (kupowane razem, podobne treściowo),
   - popularne w kategorii, trendy, nowości (cold start).
2. **Ranking** - GBDT lub sieć głęboka z cechami użytkownika, produktu i kontekstu; multi-task (P(klik), P(zakup)); wynik kalibrowany, uwzględnia cenę/marżę i stan magazynowy.
3. **Re-ranking** - różnorodność (MMR), reguły (dostępność, wykluczenie już kupionych trwałych dóbr, promocje), eksploracja.

#### Cold start

Nowy produkt: embeddingi z treści (tekst + obraz), wstrzykiwanie w sekcjach „nowości". Nowy użytkownik: popularność w kontekście, zapytania, ostatnie kliknięcia w sesji (rekomendacje sesyjne).

#### Ewaluacja

Offline: recall@k, NDCG, MAP na podziale czasowym (nie losowym, aby uniknąć wycieku). Online: A/B - CTR, konwersja, przychód na sesję, pokrycie katalogu, różnorodność; monitor negatywnych efektów (zwroty).

#### Wyzwania

Bias popularności i feedback loop, bias pozycji, sezonowość, dostępność zapasów, latencja (< 100-200 ms), spójność cech online/offline, prywatność (zgody).

**Źródła:**
- [Matrix factorization (recommender systems) (Wikipedia)](https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems))
- [Session-based Recommendations with Recurrent Neural Networks (GRU4Rec)](https://arxiv.org/abs/1511.06939)
- [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)
- [Recommender system (Wikipedia)](https://en.wikipedia.org/wiki/Recommender_system)

---

<a id="q257"></a>
### 257. Jak zbudowałbyś system wykrywania nadużyć finansowych (fraud detection)?

**Odpowiedź:**

#### Charakter problemu

- **Skrajne niezbalansowanie** (fraud to ułamek procenta transakcji).
- **Opóźnione i niepełne etykiety**: chargeback pojawia się po tygodniach; nieodkryte oszustwa siedzą jako „negatywy".
- **Adversarial**: oszuści adaptują się, więc rozkład dryfuje.
- Wymóg **niskiej latencji** (decyzja w dziesiątkach ms) i kosztów asymetrycznych: fałszywy alarm blokuje uczciwego klienta, przepuszczony fraud to strata.

#### Architektura

1. **Streaming**: transakcja -> feature service (Kafka/Flink) -> model -> decyzja (zatwierdź / odrzuć / step-up authentication np. 3DS / do ręcznej weryfikacji).
2. **Reguły + ML**: reguły dla znanych wzorców i wymogów prawnych (sankcje), model ML dla subtelnych wzorców; oba w silniku decyzyjnym.
3. **Cechy**: transakcja (kwota, kraj, sklep, MCC), użytkownik/karta (agregaty w oknach czasowych: liczba i suma transakcji w 1 min/1 h/24 h, nowe urządzenie, odległość od poprzedniej lokalizacji, prędkość), **graf** (wspólne urządzenia/adresy/karty między kontami - pierścienie oszustów, cechy GNN), urządzenie/IP, zachowanie sesji.
4. **Modele**: gradient boosting (XGBoost/LightGBM) jako główny; uzupełnienie: modele sekwencyjne, anomalie nienadzorowane (Isolation Forest, autoenkodery) dla nowych wzorców bez etykiet.

#### Niezbalansowanie

Zamiast (lub obok) resamplingu: wagi klas, dobór progu na krzywej PR pod koszt biznesowy (`koszt_FN` vs `koszt_FP`), kalibracja. Uwaga: SMOTE przed splitem powoduje wyciek.

#### Ewaluacja

PR-AUC, recall przy stałym FPR/precyzji, oczekiwana strata finansowa, ewaluacja na **podziale czasowym** (out-of-time). Monitorowanie: rozkłady cech, wskaźnik odrzuceń, wskaźnik chargebacków, drift; regularny retraining, champion/challenger, shadow mode przed wdrożeniem.

#### Pętla zwrotna

Analitycy oznaczają przypadki -> nowe etykiety; losowy niewielki ruch przepuszczany bez blokady (holdout) pozwala mierzyć rzeczywisty fraud i unikać biasu selekcji. Wyjaśnialność (SHAP) dla analityków i regulatorów.

**Źródła:**
- [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754)
- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/)
- [scikit-learn: Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874)

---

<a id="q258"></a>
### 258. Techniki fuzji multimodalnej w uczeniu maszynowym: early fusion vs late fusion

**Odpowiedź:**

#### Definicja

Uczenie multimodalne łączy informacje z kilku modalności (tekst, obraz, audio, dane tabelaryczne). **Fuzja** określa, na którym etapie i jak są łączone.

#### Early fusion (fuzja wczesna)

Cechy (lub surowe wejścia) modalności są łączone na początku - np. konkatenacja wektorów cech - i przetwarzane wspólnym modelem.
- **Zalety**: model uczy się interakcji między modalnościami na niskim poziomie, jedna sieć do trenowania, potencjalnie najlepsza dokładność, gdy modalności są ściśle powiązane i zsynchronizowane (audio + ruch warg).
- **Wady**: wymaga dopasowania (alignment) i synchronizacji, wysoka wymiarowość, jedna brakująca modalność psuje wejście, trudność, gdy modalności mają bardzo różne charakterystyki i skale, dominacja jednej modalności.

#### Late fusion (fuzja późna)

Osobny model dla każdej modalności; decyzje/logity/prawdopodobieństwa łączone na końcu (średnia, ważona suma, głosowanie, meta-klasyfikator/stacking).
- **Zalety**: modularność (można użyć najlepszych wstępnie wytrenowanych modeli per modalność), odporność na brak modalności, łatwe debugowanie i niezależne trenowanie/aktualizacje.
- **Wady**: nie modeluje interakcji między modalnościami na niskim poziomie, więc pomija zależności krzyżowe; wagi łączenia wymagają strojenia.

#### Fuzja pośrednia (intermediate / hybrid)

Kompromis: enkodery per modalność tworzą reprezentacje, które są łączone w warstwach środkowych - concatenation + MLP, **cross-attention** (tekst „patrzy" na cechy obrazu, jak w modelach VLM), bilinear pooling, gated fusion. Współczesne modele (CLIP - wspólna przestrzeń przy fuzji późnej; Flamingo, LLaVA - cross-attention/projekcje do LLM) to głównie fuzja pośrednia.

#### Kiedy co

| Sytuacja | Wybór |
|---|---|
| Mało danych, dobre modele per modalność | late fusion |
| Silne zależności czasowo-przestrzenne | early / intermediate |
| Często brakujące modalności | late fusion lub modality dropout |
| Maksymalna jakość, dużo danych | intermediate (attention) |

#### Praktyka

Stosuj **modality dropout** (losowe wyzerowanie modalności w treningu), normalizację skal cech, bądź czujny na dominację jednej modalności (np. tekstu) i na niezrównoważone tempo uczenia.

**Źródła:**
- [Multimodal Fusion Techniques in Machine Learning: Early Fusion vs Late Fusion (Amit Shekhar, LinkedIn)](https://www.linkedin.com/posts/amit-shekhar-iitbhu_machinelearning-ai-deeplearning-activity-7346410935332282368-hmoN/)
- [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406)
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020)

---

<a id="q259"></a>
### 259. Jak podejść do problemu prognozowania szeregów czasowych (time series forecasting)?

**Odpowiedź:**

#### 1. Zrozumienie problemu

- Horyzont (1 krok vs. wiele), granulacja (minuty/dni), liczba szeregów (jeden vs. tysiące), potrzeba prognozy punktowej czy przedziałowej, koszt błędu (nadprognoza vs. niedoprognoza), dostępność zmiennych zewnętrznych (święta, ceny, pogoda - znane w przyszłości czy nie).

#### 2. Eksploracja

Wykres, dekompozycja (trend + sezonowość + reszta, np. STL), stacjonarność (test ADF/KPSS), autokorelacja (ACF/PACF), braki, outliery, zmiany reżimu. Wiele szeregów: intermittent demand (dużo zer) wymaga innych metod (Croston).

#### 3. Punkty odniesienia (baseline)

Zawsze zacznij od **naive** (ostatnia wartość), **seasonal naive** (wartość sprzed sezonu), średniej kroczącej. Złożony model musi je pokonać - zaskakująco często ledwie to robi.

#### 4. Modele

- **Statystyczne**: ETS (wygładzanie wykładnicze), ARIMA/SARIMA(X), Prophet; dobre przy krótkich, pojedynczych szeregach, interpretowalne.
- **ML tabelaryczne**: gradient boosting na cechach opóźnionych (lagi, statystyki kroczące, cechy kalendarzowe, zmienne zewnętrzne); silne dla wielu szeregów (global model) i danych z regresorami.
- **Deep learning**: LSTM/GRU, N-BEATS, Temporal Fusion Transformer, PatchTST; przydatne przy dużych zbiorach wielu szeregów i wielu kowariantach; przewaga nad GBDT nie jest gwarantowana.
- Modele **probabilistyczne**: regresja kwantylowa, DeepAR, conformal prediction dla przedziałów.

#### 5. Walidacja - najczęstszy błąd

Nie używaj losowego K-fold: **wyciek z przyszłości**. Stosuj walidację z rozszerzającym/przesuwnym oknem (`TimeSeriesSplit`, rolling-origin), zachowując porządek czasu i symulując horyzont produkcyjny.

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, test_size=28)
for tr, te in tscv.split(X):
    model.fit(X[tr], y[tr]); pred = model.predict(X[te])
```

#### 6. Metryki

MAE, RMSE, MAPE/sMAPE (uwaga na zera), MASE (skalowana względem naive), pinball loss dla kwantyli, pokrycie przedziałów.

#### 7. Produkcja

Cechy dostępne w momencie prognozy (uwaga na leakage), retraining harmonogramowy, monitorowanie błędu i driftu, hierarchiczne uzgadnianie prognoz (np. sklep -> region -> całość).

**Źródła:**
- [Forecasting: Principles and Practice, 3rd ed. (Hyndman, Athanasopoulos)](https://otexts.com/fpp3/)
- [scikit-learn: TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
- [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)

---

<a id="q260"></a>
### 260. Jak zbudowałbyś system wykrywania spamu?

**Odpowiedź:**

#### Definicja i dane

Zdefiniuj spam (e-mail, komentarze, SMS, konta) i koszty błędów: **fałszywy pozytyw** (uczciwa wiadomość w spamie) zwykle kosztuje więcej niż fałszywy negatyw, więc optymalizuje się precyzję przy wysokim recallu. Dane: wiadomości z etykietami (zgłoszenia użytkowników „to spam", „to nie spam", honeypoty, ręczna weryfikacja). Zwróć uwagę na niezbalansowanie i dryf (spamerzy się adaptują).

#### Cechy

- **Tekst**: TF-IDF na słowach i n-gramach znakowych (odporne na obfuskację typu „v1agra"), embeddingi transformerów.
- **Strukturalne**: liczba linków, domeny w URL (reputacja), załączniki, proporcja wielkich liter, nagłówki (SPF/DKIM/DMARC), język, HTML vs tekst.
- **Nadawca**: reputacja IP i domeny, wiek konta, częstotliwość wysyłki, współczynnik odbić, zgłoszeń.
- **Behawioralne/sieciowe**: podobne wiadomości wysyłane do wielu odbiorców (near-duplicate, MinHash), wzorce czasowe.

#### Modele

1. **Baseline**: Naive Bayes / regresja logistyczna na TF-IDF - szybkie i zaskakująco mocne.
2. **Gradient boosting** na połączeniu cech tekstowych i metadanych.
3. **Fine-tuned transformer** (DistilBERT/BERT) dla trudniejszych przypadków - kosztowniejszy, więc jako drugi etap kaskady dla niepewnych wiadomości.
4. Reguły i blacklisty jako pierwsza, tania warstwa.

```python
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
clf = make_pipeline(TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True),
                    LogisticRegression(class_weight="balanced", max_iter=1000))
```

#### Ewaluacja

Precision/recall, PR-AUC, dobór progu przy ograniczonym FPR (np. FPR < 0,1%), podział czasowy zamiast losowego. Kalibracja i trzy strefy: blokuj / do folderu spam / przepuść.

#### Produkcja

Latencja (ms), streaming, kaskada od tańszych do droższych modeli, pętla sprzężenia zwrotnego z raportów użytkowników, częsty retraining, monitorowanie driftu i skoków spamu, ochrona przed atakami adversarialnymi (zatruwanie zgłoszeń), wyjaśnialność dla wsparcia.

**Źródła:**
- [scikit-learn: Text feature extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [scikit-learn: Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- [Naive Bayes spam filtering (Wikipedia)](https://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering)

---

<a id="q261"></a>
### 261. Opisz, jak zaimplementowałbyś system klasyfikacji obrazów

**Odpowiedź:**

#### 1. Definicja zadania i dane

Liczba klas, single-label vs multi-label, wymagane opóźnienie i platforma (chmura/edge), koszt błędów. Dane: liczba obrazów na klasę, jakość etykiet, rozkład (niezbalansowanie), zgodność z produkcją (oświetlenie, kamery). Podział train/val/test z zachowaniem grup (np. ten sam pacjent/obiekt nie może być w kilku zbiorach - leakage).

#### 2. Podejście

- **Transfer learning** jest domyślnym wyborem: model wstępnie wytrenowany na ImageNet (ResNet, EfficientNet, ConvNeXt, ViT) - najpierw trenuj tylko głowę klasyfikatora (freeze backbone), potem odmroź część warstw z małym learning rate (fine-tuning). Przy bardzo małych zbiorach: linear probe na embeddingach (np. CLIP) lub few-shot.
- Trening od zera tylko przy dużych danych i innej domenie.
- Model dla edge: MobileNet/EfficientNet-Lite, kwantyzacja INT8, distillation.

#### 3. Augmentacja i regularyzacja

Losowe przycięcia, odbicia, zmiany koloru, RandAugment/MixUp/CutMix, label smoothing, weight decay. Augmentacje muszą zachowywać etykietę (np. brak odbicia poziomego dla tekstu/cyfr).

#### 4. Trening

Strata cross-entropy (multi-label: BCE), AdamW lub SGD z momentum, scheduler (cosine, warmup), mixed precision, wczesne zatrzymanie po metryce walidacyjnej. Niezbalansowane klasy: wagi klas, focal loss, próbkowanie.

```python
import torch, torchvision
m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
for p in m.parameters(): p.requires_grad = False
for p in m.fc.parameters(): p.requires_grad = True
```

#### 5. Ewaluacja

Accuracy, macierz pomyłek, precision/recall/F1 per klasa (macro F1 przy niezbalansowaniu), top-k accuracy, kalibracja, analiza błędów (Grad-CAM), testy odporności (rotacje, szum, przesunięcie domeny).

#### 6. Wdrożenie

Eksport (TorchScript/ONNX/TensorRT), batching, kwantyzacja, monitorowanie driftu wejść i pewności, próg odrzucenia („nie wiem") dla niskiej pewności, pętla active learning z ręczną weryfikacją.

**Źródła:**
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)
- [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929)
- [EfficientNet: Rethinking Model Scaling for CNNs](https://arxiv.org/abs/1905.11946)
- [torchvision models documentation](https://pytorch.org/vision/stable/models.html)
- [Stanford CS231n: Transfer Learning](https://cs231n.github.io/transfer-learning/)

---

<a id="q262"></a>
### 262. Jakie podejście zastosowałbyś w zadaniu analizy sentymentu?

**Odpowiedź:**

#### Zdefiniuj zadanie

Poziom: dokument, zdanie czy aspekt (aspect-based: „jedzenie super, obsługa fatalna"). Etykiety: binarne, 3-klasowe (neg/neu/poz), skala 1-5, emocje. Język, domena (recenzje, tweety, finanse - słownictwo i sarkazm różnią się), wymagania latencji.

#### Progresja podejść

1. **Baseline słownikowy** (VADER, TextBlob) - bez trenowania, szybki punkt odniesienia; słaby na domenach specjalistycznych i sarkazmie.
2. **Klasyczny ML**: TF-IDF (słowa + n-gramy) + regresja logistyczna/SVM - silny baseline, tani, interpretowalny (wagi słów).
3. **Fine-tuning transformera** (BERT/RoBERTa/DistilBERT; dla polskiego HerBERT, PolBERT; wielojęzyczny XLM-R): zwykle najlepsza jakość przy kilku tysiącach etykiet. Klasyfikacja przez głowę na `[CLS]`.
4. **LLM zero/few-shot** lub z instrukcją: dobre, gdy brak etykiet, elastyczne (aspekty, uzasadnienia), ale droższe i wolniejsze; można użyć do wstępnego etykietowania i destylacji do małego modelu.

```python
from transformers import pipeline
clf = pipeline("text-classification", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
clf("Obsługa była bardzo miła, ale jedzenie zimne.")
```

#### Dane i przetwarzanie

Minimalne czyszczenie (dla transformerów zostaw interpunkcję, emoji, wielkie litery - niosą sygnał), obsługa negacji, emotikon, slangu. Jakość etykiet: kilku anotatorów, miara zgodności (kappa), jasne wytyczne dla klasy neutralnej. Niezbalansowanie: wagi klas, oversampling.

#### Ewaluacja

Macro-F1 (nie tylko accuracy), macierz pomyłek (neutralny bywa mylony), analiza błędów: sarkazm, negacje, kontekst, mieszany sentyment. Ewaluacja na danych z docelowej domeny, nie tylko na zbiorze ogólnym. Dla skali porządkowej - MAE/QWK.

#### Produkcja

Batching, distillation/kwantyzacja dla latencji, monitorowanie driftu języka (nowe memy, produkty), kalibracja, pętla ręcznej weryfikacji niepewnych przypadków.

**Źródła:**
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [Hugging Face: Text classification task guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [Sentiment analysis (Wikipedia)](https://en.wikipedia.org/wiki/Sentiment_analysis)

---

<a id="q263"></a>
### 263. Jak zaprojektowałbyś model predykcji odejścia klientów (customer churn)?

**Odpowiedź:**

#### Definicja problemu

Najpierw ustal, **co znaczy churn** i jakie okno predykcji: np. „brak aktywności/subskrypcji przez 30 dni", prognoza na 30 dni do przodu. Rozróżnij churn kontraktowy (anulowanie subskrypcji) i niekontraktowy (przestał kupować). Cel biznesowy: nie samo przewidywanie, tylko **interwencje** (oferta, kontakt), więc liczy się uplift i koszt kampanii.

#### Dane i unikanie wycieku

- Migawka na dzień T (punkt predykcji): cechy tylko z okresu przed T, etykieta = czy klient odszedł w (T, T+30].
- Cechy: demografia/plan, staż, RFM (recency/frequency/monetary), trend użycia (spadek aktywności w ostatnich tygodniach vs. wcześniej), liczba zgłoszeń do supportu, NPS, płatności nieudane, korzystanie z funkcji, zmiany planu, ceny, sygnały kontekstowe (konkurencja, sezon).
- Uwaga na leakage (cechy „po fakcie", jak anulowanie w toku). Podział **czasowy** train/val/test, nie losowy.

#### Model

- Baseline: regresja logistyczna; następnie gradient boosting (XGBoost/LightGBM) - zwykle najlepszy na tabelarycznych danych.
- **Analiza przeżycia** (Cox, Kaplan-Meier, survival GBM), gdy liczy się *kiedy* klient odejdzie i występuje cenzurowanie (klienci jeszcze aktywni).
- Niezbalansowanie: wagi klas, dobór progu, kalibracja prawdopodobieństw.

#### Ewaluacja

PR-AUC/ROC-AUC, lift/gain w top-decylu (np. „top 10% ryzyka zawiera 40% odchodzących"), kalibracja. Metryka biznesowa: oczekiwany zysk = P(churn) x skuteczność interwencji x wartość klienta - koszt oferty. Wybór progu według budżetu kampanii.

#### Od predykcji do działania

- Wyjaśnialność (SHAP) -> przyczyny churnu dla zespołu CS.
- **Uplift modeling** (kogo *interwencja* zatrzyma, a kto zostałby i tak lub odejdzie mimo oferty) + A/B z grupą kontrolną.
- Monitorowanie driftu, regularny retraining, pętla: kampania -> wynik -> nowe dane (ale uwaga: interwencje zniekształcają etykiety).

**Źródła:**
- [XGBoost documentation](https://xgboost.readthedocs.io/en/stable/)
- [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874)
- [Survival analysis (Wikipedia)](https://en.wikipedia.org/wiki/Survival_analysis)
- [Uplift modelling (Wikipedia)](https://en.wikipedia.org/wiki/Uplift_modelling)

---

<a id="q264"></a>
### 264. Jak podszedłbyś do rankingu wyników wyszukiwania?

**Odpowiedź:**

#### Architektura wielostopniowa

Ranking to kaskada, w której każdy etap jest droższy, a zbiór kandydatów mniejszy:

1. **Retrieval (recall)** - z milionów/miliardów dokumentów wybierz setki-tysiące:
   - leksykalny: BM25 na indeksie odwróconym (precyzyjny dla słów kluczowych, nazw, kodów),
   - semantyczny: bi-encoder (dense retrieval, np. DPR) + ANN (HNSW/FAISS),
   - **hybrid** (RRF lub ważona suma) - zwykle najlepszy kompromis.
2. **Ranking (precision)** - model **Learning to Rank** na top-N (setki):
   - pointwise (regresja/klasyfikacja relewancji), pairwise (RankNet), **listwise** (LambdaMART/LambdaRank - optymalizacja pod NDCG; GBDT jak LightGBM/XGBoost `rank:ndcg`),
   - cechy: BM25 na polach, cechy embeddingowe, jakość dokumentu (PageRank, autorytet), świeżość, popularność, historyczny CTR, dopasowanie zapytania, cechy personalizacji.
3. **Re-ranking** - **cross-encoder** (BERT/monoT5) przetwarza pary (zapytanie, dokument) razem, uzyskując dużo dokładniejszą relewancję, ale kosztem latencji, więc stosowany tylko do top 20-100. W RAG/wyszukiwarkach LLM - dedykowane rerankery (np. Cohere Rerank, bge-reranker).
4. **Post-processing** - różnorodność (MMR), deduplikacja, filtry bezpieczeństwa, reguły biznesowe.

#### Etykiety

Oceny ludzi (skala 0-4), sygnały niejawne z kliknięć (tanie, ale obciążone **position bias** - użyj counterfactual LTR/IPS, click models), LLM jako annotator uzupełniający. Uczenie kontrastowe z hard negatives dla retrievalu.

#### Metryki

Offline: **NDCG@k**, MRR, MAP, Recall@k (dla retrievalu). Online: CTR, czas do kliknięcia, odsetek reformulacji, abandonment, satysfakcja; A/B lub interleaving (czuły i tani).

#### Praktyczne rady

Zacznij od BM25 + LightGBM LambdaRank; dodaj retrieval gęsty, potem cross-encoder. Buduj zbiór ewaluacyjny wcześnie. Kontroluj latencję (budżet ms na etap), cache popularnych zapytań.

**Źródła:**
- [How does a Reranker work? (Outcome School)](https://outcomeschool.com/blog/how-does-a-reranker-work)
- [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906)
- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
- [ColBERT: Efficient and Effective Passage Search via Late Interaction](https://arxiv.org/abs/2004.12832)
- [Learning to rank (Wikipedia)](https://en.wikipedia.org/wiki/Learning_to_rank)

---

<a id="q265"></a>
### 265. Jak zbudowałbyś system wykrywania anomalii w ruchu sieciowym?

**Odpowiedź:**

#### Charakter problemu

Cel: wykrywać włamania, DDoS, skanowanie portów, eksfiltrację danych, malware w ruchu. Etykiet zwykle brakuje lub są nieliczne i przestarzałe, a ataki ewoluują - stąd podejście **nienadzorowane/półnadzorowane** uzupełnione regułami i (gdy są etykiety) nadzorowanym.

#### Dane i cechy

Źródła: NetFlow/IPFIX, logi zapory, DNS, pakiety (pcap), logi uwierzytelniania. Cechy z okien czasowych (np. 1 min) per host/para hostów/port: liczba bajtów i pakietów, liczba unikalnych adresów/portów docelowych, rozkład rozmiarów pakietów, flagi TCP, entropia adresów, odsetek błędów, częstotliwości żądań DNS, geolokalizacja. Model bazowy uwzględnia sezonowość (dzień/tydzień).

#### Metody

1. **Statystyczne/progowe**: z-score, MAD, EWMA, CUSUM na szeregach czasowych wolumenu; szybkie i interpretowalne, dobre dla DDoS.
2. **Klasyczne ML nienadzorowane**: Isolation Forest, Local Outlier Factor, One-Class SVM, klasteryzacja (DBSCAN).
3. **Głębokie**: autoenkoder / VAE (wysoki błąd rekonstrukcji = anomalia), LSTM/Transformer prognozujący sekwencję i oceniający odchylenie, modele grafowe (komunikacja host-host).
4. **Nadzorowane** (gdy są etykiety, np. z IDS/SOC): gradient boosting/sieć na znanych atakach; nie wykryje nowych typów.
5. **Reguły/sygnatury** (Snort/Suricata) jako uzupełnienie.

```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(n_estimators=300, contamination="auto", random_state=0).fit(X_normal)
score = -iso.score_samples(X_new)   # wyższy = bardziej anomalny
```

#### Architektura

Streaming (Kafka/Flink) -> agregacja cech w oknach -> scoring -> deduplikacja i korelacja alertów -> priorytetyzacja -> SOC (SIEM). Trenuj na okresie „normalnym", reguluj próg przez dopuszczalną liczbę alertów dziennie.

#### Ewaluacja

Brak etykiet: wstrzykiwanie syntetycznych ataków (red team), ręczna weryfikacja próbek, precyzja alertów (analitycy oznaczają), czas wykrycia, odsetek fałszywych alarmów (alert fatigue jest głównym problemem). Z etykietami: PR-AUC, recall dla ataków, benchmark (CIC-IDS, UNSW-NB15) - ale uwaga na nierealność zbiorów.

#### Wyzwania

Concept drift (nowe usługi, zmiana ruchu), szyfrowanie (tylko metadane), adversarial evasion, skala, wyjaśnialność alertów.

**Źródła:**
- [scikit-learn: Novelty and Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Deep Learning for Anomaly Detection: A Survey](https://arxiv.org/abs/1901.03407)
- [Anomaly detection (Wikipedia)](https://en.wikipedia.org/wiki/Anomaly_detection)

---

<a id="q266"></a>
### 266. Jak wybrać właściwy algorytm uczenia maszynowego?

**Odpowiedź:**

Nie ma jednego „najlepszego" algorytmu (twierdzenie no free lunch). Wybór jest procesem, nie odruchem.

#### 1. Rodzaj problemu

- Nadzorowane: klasyfikacja / regresja / ranking; nienadzorowane: klasteryzacja, redukcja wymiaru, anomalie; sekwencje/szeregi czasowe; RL. Rodzaj danych rozstrzyga o rodzinie modeli: tabelaryczne -> drzewa; obrazy -> CNN/ViT; tekst -> transformery; grafy -> GNN.

#### 2. Charakterystyka danych

| Czynnik | Wpływ na wybór |
|---|---|
| Mało danych (setki-tysiące) | modele proste (regresja, SVM, drzewa), transfer learning |
| Dużo danych, dane nieustrukturyzowane | deep learning |
| Dane tabelaryczne | gradient boosting (XGBoost/LightGBM/CatBoost) zwykle wygrywa z sieciami |
| Wiele cech, rzadkie (tekst BoW) | modele liniowe z regularyzacją |
| Braki i cechy kategoryczne | CatBoost/LightGBM ograniczają preprocessing |
| Silna nieliniowość i interakcje | drzewa, sieci, kernel SVM |

#### 3. Wymagania niefunkcjonalne

Interpretowalność (regulacje, medycyna: regresja logistyczna, drzewa, GAM, SHAP), latencja i pamięć predykcji (edge, ms), koszt i czas treningu, możliwość uczenia online, odporność na outliery, skalowalność.

#### 4. Procedura

1. Zdefiniuj metrykę biznesową i schemat walidacji (właściwy podział: czasowy/grupowy).
2. Zbuduj **prosty baseline** (średnia, regresja logistyczna, drzewo) - wyznacza, ile warta jest złożoność.
3. Wypróbuj kilka rodzin (liniowa, GBDT, sieć) z podstawowym tuningiem; porównuj na tej samej walidacji krzyżowej z odchyleniem standardowym, nie jedną liczbą.
4. Dostrój hiperparametry najlepszych (Optuna/random search), rozważ ensemble.
5. Oceń kompromisy: dokładność vs. interpretowalność vs. koszt wdrożenia; zweryfikuj na zbiorze testowym raz.

#### Wskazówki

- Dla tabelarycznych: zacznij od GBDT i regresji logistycznej (Grinsztajn i in. pokazują, że modele drzewiaste często dominują na średnich zbiorach tabelarycznych).
- Nie dostrajaj skrajnie, jeśli różnice mieszczą się w szumie.
- Prosty model, który da się utrzymać, bije złożony, który się psuje.

**Źródła:**
- [scikit-learn: Choosing the right estimator](https://scikit-learn.org/stable/machine_learning_map.html)
- [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815)
- [Google: Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [No free lunch theorem (Wikipedia)](https://en.wikipedia.org/wiki/No_free_lunch_theorem)

---

<a id="q267"></a>
### 267. Czym jest dryf modelu (model drift) i jak sobie z nim radzić?

**Odpowiedź:**

#### Definicja

**Model drift** (degradacja modelu) to spadek jakości predykcji w czasie, bo świat produkcyjny różni się od danych treningowych. Główne rodzaje:

- **Data drift / covariate shift** - zmienia się rozkład wejść `P(X)`, a zależność `P(y|X)` pozostaje (np. nowa grupa klientów, nowy sensor).
- **Concept drift** - zmienia się relacja `P(y|X)` (np. zmiana zachowań oszustów, kryzys ekonomiczny zmienia wzorce kredytowe). Może być nagły, stopniowy, sezonowy, cykliczny.
- **Label / prior shift** - zmienia się `P(y)` (więcej fraudów niż wcześniej).
- **Upstream/data pipeline drift** - zmiana schematu, jednostek, sposobu logowania, błąd ETL (często to prawdziwa przyczyna).

#### Wykrywanie

- Gdy etykiety są dostępne szybko: bezpośrednio monitoruj metryki jakości (accuracy, AUC, MAE) w oknach czasowych.
- Gdy etykiety są opóźnione (typowe): monitoruj **proxy** - rozkład wejść i predykcji.
   - testy: Kolmogorov-Smirnov (cechy ciągłe), chi-kwadrat (kategoryczne), **PSI** (Population Stability Index; reguła praktyczna: < 0,1 stabilnie, > 0,25 duża zmiana), Jensen-Shannon/Wasserstein, klasyfikator odróżniający dane treningowe od bieżących (domain classifier),
   - monitoruj rozkład predykcji, pewności, odsetek braków, wartości spoza zakresu, ważność cech.
- Alerty z progami i oknem referencyjnym; uwaga na sezonowość (porównuj do tego samego okresu).

```python
from scipy.stats import ks_2samp
stat, p = ks_2samp(train_df["age"], live_df["age"])
```

#### Reakcja

1. **Diagnoza**: pipeline vs. rzeczywisty drift; segmenty, w których jakość spada.
2. **Retraining** na świeżych danych - harmonogramowy (np. co tydzień) lub wyzwalany alertem; okno przesuwne lub ważenie nowszych próbek.
3. **Uczenie online/przyrostowe** tam, gdzie zmiany są szybkie.
4. Cechy odporniejsze na drift, unikanie cech kruchych (bezwzględne ceny -> znormalizowane).
5. Fallback (prostszy model/reguły), champion-challenger, shadow deployment, automatyczna walidacja przed wdrożeniem, rollback.
6. Zapewnij pętlę etykiet (ręczna weryfikacja próbek, ground truth z opóźnieniem).

#### Pułapki

Retraining na własnych decyzjach modelu (feedback loop), nadmierne alerty (zmęczenie), reagowanie na sezonowość jak na drift.

**Źródła:**
- [Concept drift (Wikipedia)](https://en.wikipedia.org/wiki/Concept_drift)
- [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953)
- [Evidently AI documentation (data drift)](https://docs.evidentlyai.com/)
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)

---

<a id="q268"></a>
### 268. Jak poradzisz sobie z trenowaniem na danych wielkoskalowych (large-scale data)?

**Odpowiedź:**

#### Diagnoza wąskiego gardła

Najpierw sprawdź, co ogranicza: **pamięć** (dane/model nie mieszczą się w RAM/GPU), **I/O** (dysk/sieć nie nadąża za GPU), **obliczenia** (za wolno) czy **przygotowanie danych** (CPU preprocessing). Profiluj przed optymalizacją.

#### Dane

- **Formaty kolumnowe i binarne**: Parquet/Arrow zamiast CSV; dla obrazów/tekstu - sharded formaty (WebDataset, TFRecord, MosaicML Streaming), tokeny w plikach binarnych (memmap).
- **Streaming** i lazy loading: `IterableDataset`, Hugging Face `datasets` w trybie `streaming=True` - bez ładowania całości.
- **Rozproszone przetwarzanie**: Spark, Dask, Ray Data do ETL i feature engineeringu; feature store.
- **DataLoader**: `num_workers`, `prefetch_factor`, `pin_memory`, unikanie kosztownego preprocessingu w pętli (prekompilacja cech/tokenizacja offline).
- **Próbkowanie i deduplikacja**: często lepiej zbudować dobrze dobrany, czysty podzbiór; dla eksperymentów stratyfikowana próbka; dedup zwiększa jakość i zmniejsza koszt.

#### Trening

- **Mini-batch SGD** i uczenie przyrostowe (`partial_fit` w sklearn dla SGDClassifier, Naive Bayes) - dane przetwarzane porcjami.
- **Modele skalowalne**: liniowe z SGD, GBDT rozproszone (XGBoost/LightGBM na Spark/Dask), hashing trick dla cech kategorycznych o wysokiej kardynalności.
- **Równoległość na wielu GPU/węzłach**: data parallelism (`DistributedDataParallel`), model/tensor/pipeline parallelism dla dużych modeli, **FSDP/ZeRO** dzielące parametry, gradienty i stan optymalizatora.
- Mixed precision (bf16), gradient accumulation (duży efektywny batch), gradient checkpointing (pamięć kosztem obliczeń).
- Checkpointy i wznawianie treningu (awarie na dużym klastrze są normą).

```python
# PyTorch DDP: uruchom torchrun --nproc_per_node=8 train.py
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
```

#### Uwagi praktyczne

Skalowanie learning rate z rozmiarem batcha (liniowo + warmup), monitorowanie przepustowości (samples/s) i utylizacji GPU, powtarzalność (seedy, shardy), koszt vs. zysk (zwykle malejące zwroty od ilości danych - sprawdź krzywą uczenia).

**Źródła:**
- [PyTorch: Distributed Data Parallel](https://pytorch.org/docs/stable/notes/ddp.html)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [Hugging Face Datasets: Stream](https://huggingface.co/docs/datasets/stream)
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)

---

<a id="q269"></a>
### 269. Jak radzisz sobie z zaszumionymi danymi (noisy data) w modelach ML?

**Odpowiedź:**

#### Rodzaje szumu

- **Szum w cechach** (błędy pomiaru, outliery, literówki).
- **Szum w etykietach** (błędy anotatorów, niejednoznaczność, automatyczne etykiety) - zwykle groźniejszy, bo model potrafi go zapamiętać.
- Szum losowy vs. systematyczny (systematyczny = bias, nie znika wraz z ilością danych).

#### Diagnoza

Rozkłady, wykresy, reguły walidacyjne (zakresy, typy), analiza outlierów (IQR, z-score, Isolation Forest), porównanie anotatorów (inter-annotator agreement), analiza największych błędów modelu (często to błędne etykiety), krzywe uczenia (duży rozjazd train/val sugeruje overfitting szumu).

#### Metody dla cech

- Czyszczenie i walidacja na wejściu pipeline'u; imputacja braków; wygładzanie szeregów (średnia krocząca, filtr Kalmana).
- **Winsoryzacja**/przycinanie, transformacje (log), skalowanie odporne (`RobustScaler`, mediana i IQR).
- Redukcja wymiaru (PCA), selekcja cech - usunięcie cech nieinformatywnych.
- Uśrednianie wielu pomiarów.

#### Metody dla etykiet

- **Confident learning** (cleanlab): wykrywa prawdopodobnie błędne etykiety na podstawie predykcji out-of-fold, po czym można je poprawić lub usunąć.
- Poprawa procesu: wiele oceniających + głosowanie/Dawid-Skene, lepsze instrukcje, przegląd przypadków spornych.
- **Odporne funkcje straty**: MAE zamiast MSE (regresja), Huber, symetryczna cross-entropy, generalized cross-entropy, **label smoothing**.
- Metody deep learningu: co-teaching (dwie sieci wymieniają „małe-loss" próbki), early stopping (sieci najpierw uczą się czystych wzorców, potem zapamiętują szum), MixUp.

#### Ogólne strategie

- Regularyzacja (L2, dropout, augmentacja) i modele o mniejszej pojemności.
- **Ensemble/bagging** (Random Forest) redukuje wariancję wywołaną szumem.
- Więcej danych rozcieńcza szum losowy (nie systematyczny).
- Czysty, mały zbiór walidacyjny/testowy - abyś mierzył prawdziwą jakość.

#### Uwaga

Nie usuwaj „outlierów" bezmyślnie - mogą być rzadkimi, ważnymi zdarzeniami (fraud, awarie). Zawsze potwierdzaj metodą i domeną.

**Źródła:**
- [Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/abs/1911.00068)
- [Learning from Noisy Labels with Deep Neural Networks: A Survey](https://arxiv.org/abs/2007.08199)
- [Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels](https://arxiv.org/abs/1804.06872)
- [scikit-learn: Preprocessing data (RobustScaler)](https://scikit-learn.org/stable/modules/preprocessing.html)

---

<a id="q270"></a>
### 270. Jakie strategie zastosujesz, by skrócić czas trenowania modelu deep learningowego?

**Odpowiedź:**

#### Zasada nadrzędna

**Najpierw profiluj** (PyTorch Profiler, `nvidia-smi`, przepustowość samples/s), żeby wiedzieć, czy wąskim gardłem jest GPU, dane (I/O, CPU) czy komunikacja. Bez tego optymalizacje bywają bezcelowe.

#### Dane i I/O

- Więcej workerów DataLoadera, `pin_memory=True`, `prefetch_factor`, `persistent_workers=True`.
- Szybkie formaty (sharded, binarne), dysk NVMe, prekompilacja/tokenizacja offline, augmentacje na GPU (DALI, Kornia).
- GPU nie może czekać na dane: utylizacja ~100% to cel.

#### Obliczenia

- **Mixed precision** (fp16/bf16 z `torch.autocast`, GradScaler dla fp16): zwykle kilkukrotne przyspieszenie na nowoczesnych GPU i mniejsze zużycie pamięci.
- **`torch.compile`**, fused optimizery (`fused=True`), operacje Tensor Core (wymiary jako wielokrotności 8), `channels_last` dla CNN.
- **Efektywne uwagi**: FlashAttention / `scaled_dot_product_attention` - mniej pamięci i szybciej przy długich sekwencjach.
- Większy batch (do granic pamięci) + skalowanie learning rate, gradient accumulation.

```python
model = torch.compile(model)
scaler = torch.amp.GradScaler()
with torch.autocast("cuda", dtype=torch.bfloat16):
    loss = model(x).loss
```

#### Skala i równoległość

- Multi-GPU: `DistributedDataParallel`, FSDP/DeepSpeed dla dużych modeli; komunikacja NCCL; gradient checkpointing, gdy brakuje pamięci (wolniej per krok, ale pozwala na większe batche).

#### Algorytmicznie (mniej kroków)

- **Transfer learning / fine-tuning** zamiast treningu od zera; parameter-efficient (LoRA) przy dużych modelach.
- Dobry optymalizator i harmonogram: AdamW, warmup + cosine, dobrze dobrany learning rate (LR finder) - często największy zysk; super-convergence (one-cycle).
- Wczesne zatrzymanie, mniejszy model lub rozdzielczość na początku (progressive resizing), przycięcie zbioru (dedup, curriculum, próbkowanie informatywnych przykładów).
- Distillation, mniejsza architektura o podobnej jakości.
- Szybsze eksperymenty: mały podzbiór i model do wyboru hiperparametrów, wczesne przerywanie słabych prób (Hyperband/ASHA).

#### Weryfikacja

Mierz czas do docelowej jakości (time-to-accuracy), nie tylko czas kroku; upewnij się, że mixed precision i większy batch nie pogarszają zbieżności.

**Źródła:**
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [PyTorch: torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

---

<a id="q271"></a>
### 271. Jak wdrożyć model ML na produkcję?

**Odpowiedź:**

#### 1. Przygotowanie modelu

- Powtarzalny trening (kod, dane, konfiguracja i seedy wersjonowane; DVC/MLflow), rejestr modeli (model registry) z metadanymi, metrykami i lineage.
- Wymagania: latencja, przepustowość, koszt, pamięć - wpływają na architekturę (kwantyzacja, distillation, ONNX/TensorRT).
- Testy: jednostkowe dla pipeline'u cech, testy walidacji danych, testy jakości na zbiorze złotym, testy odporności i sprawiedliwości.

#### 2. Wzorce serwowania

| Wzorzec | Kiedy | Uwagi |
|---|---|---|
| **Batch (offline)** | rekomendacje dzienne, scoring nocny | prosty, tani; wyniki w bazie |
| **Online (real-time API)** | decyzje w ms (fraud, ranking) | REST/gRPC, autoskalowanie, cache |
| **Streaming** | zdarzenia ciągłe | Kafka/Flink + model |
| **Edge / on-device** | prywatność, brak sieci | kwantyzacja, TFLite/Core ML |

Narzędzia: FastAPI + kontener, TorchServe, Triton Inference Server, KServe/Seldon (Kubernetes), BentoML, dla LLM vLLM/TGI, usługi zarządzane (SageMaker, Vertex AI).

```python
from fastapi import FastAPI
app = FastAPI()
@app.post("/predict")
def predict(x: Features):
    return {"score": float(model.predict_proba([x.to_vec()])[0, 1])}
```

#### 3. Infrastruktura

Kontenery (Docker) z zablokowanymi zależnościami, orkiestracja Kubernetes, autoskalowanie (HPA po latencji/QPS), GPU vs. CPU, dynamic batching, cache predykcji/cech. **Feature store** zapewnia spójność cech offline/online (unika training-serving skew).

#### 4. CI/CD dla ML

Pipeline: walidacja danych -> trening -> ewaluacja względem aktualnego modelu (bramka jakości) -> rejestracja -> wdrożenie. Automatyzacja (GitHub Actions, Airflow, Kubeflow).

#### 5. Bezpieczne wdrażanie

- **Shadow deployment** (nowy model liczy równolegle, bez wpływu),
- **Canary** (mały odsetek ruchu) i **A/B test**,
- **blue/green** i szybki **rollback**,
- champion-challenger.

#### 6. Po wdrożeniu

Monitoring (jakość, drift, latencja, błędy), logowanie predykcji i cech, pętla etykiet, zaplanowany retraining, bezpieczeństwo (uwierzytelnienie, limity, ochrona danych), dokumentacja (model card).

**Źródła:**
- [Google: Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- [MLflow documentation](https://mlflow.org/docs/latest/index.html)
- [NVIDIA Triton Inference Server documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
- [Designing Machine Learning Systems (Chip Huyen)](https://huyenchip.com/machine-learning-systems-design/toc.html)

---

<a id="q272"></a>
### 272. Jak monitorować wydajność modelu na produkcji?

**Odpowiedź:**

#### Cztery warstwy monitoringu

1. **Infrastruktura i serwis**: latencja (p50/p95/p99), przepustowość (QPS), błędy 5xx, timeouty, zużycie CPU/GPU/pamięci, długość kolejek, koszt. Narzędzia: Prometheus + Grafana, Datadog, OpenTelemetry.
2. **Jakość danych wejściowych**: odsetek braków, typy i zakresy, nowe kategorie, schemat, świeżość danych, spójność cech online/offline. Wykrywa awarie pipeline'u (najczęstsza przyczyna incydentów).
3. **Drift danych i predykcji**: rozkłady cech i wyników modelu względem referencji (PSI, KS, chi-kwadrat, JS divergence), rozkład pewności, wskaźnik odrzuceń/pozytywów.
4. **Jakość modelu (metryki biznesowe i ML)**: accuracy/precision/recall/AUC/MAE w oknach czasowych, per segment (kraj, urządzenie, klient), kalibracja, metryki biznesowe (konwersja, przychód, odsetek skarg).

#### Problem opóźnionych etykiet

Ground truth często przychodzi z opóźnieniem (chargeback po tygodniach) lub wcale. Rozwiązania: podłączanie etykiet, gdy dotrą (join po ID predykcji) i liczenie metryk wstecz; **proxy metrics** (drift, zmiana rozkładu predykcji, CTR); ręczna ocena próbek (active sampling), holdout losowego ruchu, LLM-judge w GenAI; szacowanie wydajności bez etykiet (np. CBPE w NannyML).

#### Implementacja

- Loguj każdą predykcję: ID, timestamp, wersja modelu, cechy wejściowe (lub hash/próbka, z uwzględnieniem prywatności), wynik, decyzja.
- Zaplanowane joby liczące metryki i drift (godziny/dni); dashboard z trendami, **alerty** z progami (statyczne i adaptacyjne, uwaga na sezonowość) kierowane do właściwego zespołu z runbookiem.
- Segmentacja: średnia potrafi ukryć degradację w podgrupie.
- Wersjonowanie i porównanie champion vs. challenger, canary z automatycznym rollbackiem.
- Dla LLM: monitoruj też koszt tokenów, halucynacje/faithfulness, odsetek odmów, feedback użytkowników, jakość retrievalu, prompt injection.

```python
# Evidently: raport driftu
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
Report(metrics=[DataDriftPreset()]).run(reference_data=ref, current_data=cur)
```

#### Reakcja

Alert -> triage (pipeline vs. prawdziwy drift) -> naprawa danych / retraining / rollback -> post-mortem. Zdefiniuj SLO dla modelu (np. AUC > X, p95 < Y ms) i właściciela.

**Źródła:**
- [Evidently AI documentation](https://docs.evidentlyai.com/)
- [Prometheus documentation: Overview](https://prometheus.io/docs/introduction/overview/)
- [Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift](https://arxiv.org/abs/1810.11953)
- [Google: Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)

---

<a id="q273"></a>
### 273. Jak wdrożyć model o rygorystycznych wymaganiach dotyczących opóźnień (low-latency)?

**Odpowiedź:**

Przy wdrożeniach low-latency (np. rekomendacje, wykrywanie oszustw, autouzupełnianie) liczy się zwykle **p95/p99 latency**, a nie średnia. Najpierw ustalamy budżet czasu (SLO) dla całego żądania: sieć + pobranie cech + preprocessing + inferencja + postprocessing.

#### Optymalizacja samego modelu
- **Mniejszy model / architektura**: distillation (uczeń naśladuje większego nauczyciela), pruning, tańsze architektury (MobileNet, DistilBERT).
- **Quantization**: FP32 -> FP16/INT8 zmniejsza pamięć i przyspiesza inferencję (zwykle kosztem niewielkiego spadku jakości; wymaga walidacji).
- **Kompilacja i runtime**: ONNX Runtime, TensorRT, `torch.compile`, XLA; fuzja operatorów, optymalizacja grafu.
- **Sprzęt**: GPU/TPU/akceleratory dla dużych modeli, dobrze dostrojone CPU dla małych (modele drzewiaste, regresja).

#### Optymalizacja serwowania
- **Dynamic batching** (Triton, TorchServe): grupowanie żądań podnosi throughput, ale dodaje opóźnienie kolejki — trzeba dobrać maksymalny czas oczekiwania.
- **Warm-up** i utrzymywanie modeli w pamięci (brak cold startów), autoscaling z zapasem.
- **Caching** wyników dla powtarzalnych wejść oraz embeddingów.
- **Feature store online** (Redis, DynamoDB) z pre-computed cechami zamiast liczenia ich w locie.
- **Cascade / early exit**: tani model obsługuje łatwe przypadki, drogi tylko trudne.
- **Wydajny protokół**: gRPC zamiast REST/JSON, kolokacja serwisów w tym samym regionie, edge/on-device.
- **Async / precompute**: jeśli wynik może być obliczony offline (np. rekomendacje wsadowe), serwujemy go z cache.

#### Pomiar i pułapki
- Profiluj (PyTorch Profiler, Nsight), mierz end-to-end, a nie tylko `model.forward`.
- Testuj pod realnym obciążeniem (load testing), monitoruj percentyle.
- Zawsze weryfikuj, że optymalizacja nie pogorszyła metryki jakości (shadow deployment, A/B).

**Źródła:**
- [NVIDIA Triton Inference Server](https://developer.nvidia.com/triton-inference-server)
- [ONNX Runtime — dokumentacja](https://onnxruntime.ai/docs/)
- [Rules of Machine Learning (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Distilling the Knowledge in a Neural Network (Hinton i in.)](https://arxiv.org/abs/1503.02531)

---

<a id="q274"></a>
### 274. Jakie są typowe wyzwania przy wdrażaniu modeli ML?

**Odpowiedź:**

Wdrożenie to zwykle trudniejsza część projektu niż samo trenowanie; w słynnej pracy o „ukrytym długu technicznym" pokazano, że sam kod modelu to niewielki fragment całego systemu ML.

#### Wyzwania danych
- **Training-serving skew**: różnice między cechami w treningu a w produkcji (inny kod preprocessingu, inne źródła danych, wycieki czasowe). Remedium: wspólny kod/feature store, testy spójności.
- **Data drift i concept drift**: rozkład wejść (`P(X)`) lub zależność `P(y|X)` zmienia się z czasem; jakość modelu spada po cichu.
- **Jakość danych**: brakujące wartości, zmiany schematu, opóźnione dane.

#### Wyzwania inżynieryjne
- **Latency i koszt** (GPU, autoscaling), skalowalność, niezawodność.
- **Reprodukowalność**: wersje danych, kodu, zależności, hiperparametrów, seedów; kontenery (Docker).
- **Integracja** z istniejącymi systemami, kontrakty API, wersjonowanie modeli.
- **Bezpieczne rollouty**: shadow mode, canary, A/B, szybki rollback.
- **Brak etykiet w produkcji** (opóźniony ground truth) — trudność w mierzeniu jakości na bieżąco.

#### Wyzwania organizacyjne i etyczne
- Współpraca data science / inżynieria / produkt, brak własności (ownership) po wdrożeniu.
- **Explainability, fairness, prywatność, zgodność** (RODO, regulacje sektorowe).
- **Bezpieczeństwo**: adversarial inputs, kradzież modelu, wycieki danych.

#### Jak sobie radzić
CI/CD/CT (continuous training), monitoring danych i predykcji, alerty, automatyczne retraining, testy (dane, model, infrastruktura), model registry, dokumentacja (model cards).

**Źródła:**
- [Hidden Technical Debt in Machine Learning Systems (NeurIPS 2015)](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)
- [Rules of Machine Learning (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [MLOps: Continuous delivery and automation pipelines (Google Cloud)](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

---

<a id="q275"></a>
### 275. Omów wymagania dotyczące skalowalności i opóźnień w systemach ML.

**Odpowiedź:**

Dwie podstawowe miary wydajności serwowania to:
- **Latency** — czas odpowiedzi na pojedyncze żądanie (zwykle raportowany jako p50/p95/p99).
- **Throughput** — liczba żądań (lub próbek) na sekundę (QPS).

Są w napięciu: batching zwiększa throughput i wykorzystanie GPU, ale wydłuża latency pojedynczego żądania.

#### Skalowalność
- **Pozioma** (więcej replik za load balancerem, Kubernetes HPA) vs **pionowa** (mocniejsza maszyna).
- **Sharding modelu** (tensor/pipeline parallelism) dla modeli, które nie mieszczą się na jednym urządzeniu (LLM).
- **Skalowanie treningu**: data parallelism, model parallelism, mixed precision.
- **Skalowanie danych**: rozproszone przetwarzanie (Spark, Beam), formaty kolumnowe (Parquet).

#### Wzorce architektoniczne
| Tryb | Opis | Kiedy |
|---|---|---|
| Batch | predykcje offline, zapis do bazy | rekomendacje dzienne, scoring |
| Online (real-time) | predykcja na żądanie | fraud, wyszukiwanie |
| Streaming | predykcja na strumieniu zdarzeń (Kafka/Flink) | monitoring, IoT |
| Edge / on-device | inferencja lokalnie | prywatność, brak sieci |

#### Kompromisy i planowanie
- Koszt vs jakość: większy model = lepsza jakość, ale wyższa latency i koszt.
- Kaskady modeli, caching, kompresja modelu.
- Planowanie pojemności: peak vs średni ruch, autoscaling, kolejki, backpressure, degradacja łagodna (fallback do prostszego modelu).
- Definiujemy SLO/SLA i monitorujemy je; testy obciążeniowe przed wdrożeniem.

**Źródła:**
- [Designing Machine Learning Systems (Chip Huyen) — strona książki](https://huyenchip.com/books/)
- [NVIDIA Triton — dokumentacja](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
- [Kubernetes — Horizontal Pod Autoscaling](https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/)

---

<a id="q276"></a>
### 276. Jak zapewnić, że model jest skalowalny i dobrze działa na dużych zbiorach danych?

**Odpowiedź:**

Skalowalność dotyczy zarówno **treningu**, jak i **inferencji** oraz całego pipeline'u danych.

#### Dane i pipeline
- Przetwarzanie rozproszone (Spark, Dask, Ray, Beam), formaty kolumnowe (Parquet), partycjonowanie.
- **Streaming danych** z dysku zamiast ładowania do RAM (`torch.utils.data.DataLoader` z wieloma workerami, `tf.data`, Hugging Face `datasets` w trybie streaming).
- Wydajny preprocessing: wektoryzacja, cache'owanie, prefetch.

#### Algorytmy
- Metody online/inkrementalne: `SGDClassifier.partial_fit`, mini-batch k-means, `IncrementalPCA`.
- Przybliżenia: hashing trick, aproksymowane najbliższe sąsiedztwo (FAISS, HNSW), sampling, redukcja wymiarowości.
- Modele wspierające duże dane: LightGBM/XGBoost (histogramowe), rozproszone GBDT.

#### Trening rozproszony
- **Data parallelism** (DDP), **model/pipeline/tensor parallelism**, ZeRO/FSDP dla dużych modeli.
- **Mixed precision** (FP16/BF16), gradient accumulation, gradient checkpointing.
- Skalowanie learning rate wraz z rozmiarem batcha (warm-up).

#### Weryfikacja jakości przy skali
- Więcej danych nie zawsze pomaga: sprawdź krzywe uczenia (learning curves), aby ocenić, czy jesteśmy w reżimie wysokiego biasu czy wariancji.
- Zadbaj o jakość etykiet i deduplikację — przy dużej skali szum i duplikaty psują metryki.
- Podział walidacyjny bez wycieków (np. wg czasu lub użytkownika).
- Monitoruj koszt (czas/$ na epokę) i throughput.

```python
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(loss="log_loss")
for X_batch, y_batch in stream_batches():
    clf.partial_fit(X_batch, y_batch, classes=[0, 1])
```

**Źródła:**
- [scikit-learn: Strategies to scale computationally — bigger data](https://scikit-learn.org/stable/computing/scaling_strategies.html)
- [PyTorch — Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [FAISS — repozytorium](https://github.com/facebookresearch/faiss)

---

<a id="q277"></a>
### 277. Czym jest explainability modelu (wyjaśnialność)? Dlaczego jest ważna?

**Odpowiedź:**

**Explainability** (wyjaśnialność) to zdolność do przedstawienia w zrozumiały dla człowieka sposób, *dlaczego* model podjął daną decyzję. Blisko związana jest **interpretability** (interpretowalność) — stopień, w jakim człowiek potrafi zrozumieć mechanizm modelu (np. regresja liniowa czy małe drzewo). Zwykle rozróżnia się modele **interpretowalne z natury** (glass-box) i **wyjaśnianie post-hoc** modeli czarnoskrzynkowych (black-box).

#### Zakres wyjaśnień
- **Globalne**: jak model działa ogółem (ważność cech, ogólne zależności).
- **Lokalne**: dlaczego konkretna predykcja wyszła tak, a nie inaczej.

#### Dlaczego to ważne
- **Zaufanie i adopcja** przez użytkowników i biznes.
- **Debugowanie**: wykrywanie data leakage, spurious correlations (np. klasyfikator wykrywa tło zamiast obiektu), błędów w danych.
- **Regulacje i zgodność**: RODO (prawo do informacji o logice decyzji), sektor finansowy i medyczny, wymogi audytowe.
- **Fairness**: wykrycie uprzedzeń i wykorzystywania cech wrażliwych lub ich proxy.
- **Bezpieczeństwo i odpowiedzialność** w zastosowaniach wysokiego ryzyka.
- **Wiedza o dziedzinie**: model może ujawnić nowe zależności.

#### Ograniczenia
- Wyjaśnienia post-hoc są przybliżeniami; mogą być niestabilne lub mylące (np. korelowane cechy w SHAP/permutation importance).
- Istnieje kompromis (nie zawsze) między dokładnością a interpretowalnością.
- Wyjaśnienie „wiarygodne" dla człowieka nie musi być wiernym opisem modelu (faithfulness).

**Źródła:**
- [Interpretable Machine Learning (Christoph Molnar)](https://christophm.github.io/interpretable-ml-book/)
- [Towards A Rigorous Science of Interpretable Machine Learning (Doshi-Velez, Kim)](https://arxiv.org/abs/1702.08608)
- [Stop Explaining Black Box Machine Learning Models... (Rudin)](https://arxiv.org/abs/1811.10154)

---

<a id="q278"></a>
### 278. Jakich technik użyłbyś, aby model był bardziej interpretowalny?

**Odpowiedź:**

Podejścia dzielimy na dwie grupy.

#### 1. Modele interpretowalne z założenia
- **Regresja liniowa/logistyczna** (z regularyzacją L1 dla rzadkich, czytelnych współczynników).
- **Płytkie drzewa decyzyjne**, listy reguł (rule lists), **GAM** i **Explainable Boosting Machine (EBM)**.
- **Ograniczenia monotoniczności** (monotonic constraints w XGBoost/LightGBM), sparse modele, scorecards.
- Sieci z wbudowaną interpretowalnością: attention (ostrożnie!), concept bottleneck models.

#### 2. Wyjaśnienia post-hoc
- **Feature importance**: impurity-based (obciążone), **permutation importance** (`sklearn.inspection.permutation_importance`).
- **PDP / ICE / ALE**: wpływ cechy na predykcję (PDP zakłada niezależność cech; ALE lepiej znosi korelacje).
- **LIME**: lokalny, prosty model zastępczy wokół predykcji.
- **SHAP**: wartości Shapleya — addytywny, spójny podział wkładu cech; `TreeExplainer` jest szybki dla drzew.
- **Sieci neuronowe**: saliency maps, Integrated Gradients, Grad-CAM (obrazy), analiza attention/probing (NLP).
- **Counterfactual explanations**: „co należałoby zmienić, aby wynik był inny".
- **Surrogate models**: prosty model naśladujący złożony (np. drzewo trenowane na predykcjach czarnej skrzynki).
- **Przykłady**: prototypy, najbliżsi sąsiedzi, influence functions.

#### Praktyka
- Ogranicz liczbę i złożoność cech; nadaj im biznesowe znaczenie.
- Sprawdzaj stabilność wyjaśnień i wpływ skorelowanych cech.
- Zaczynaj od prostego baseline'u — jeśli osiąga on zbliżoną jakość, nie ma potrzeby czarnej skrzynki.
- Dokumentuj (model cards).

```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values)
```

**Źródła:**
- [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874)
- ["Why Should I Trust You?" (LIME)](https://arxiv.org/abs/1602.04938)
- [Grad-CAM](https://arxiv.org/abs/1610.02391)
- [scikit-learn: Inspection (PDP, permutation importance)](https://scikit-learn.org/stable/inspection.html)

---

<a id="q279"></a>
### 279. Opisz swoje podejście do debugowania niedziałającego (słabo działającego) modelu ML.

**Odpowiedź:**

Debugowanie ML robimy **systematycznie, od najprostszych i najtańszych hipotez**, zmieniając jedną rzecz naraz.

#### 1. Sprawdź dane i pipeline
- Poprawność etykiet, duplikaty, brakujące wartości, jednostki, rozkłady; wizualna inspekcja próbek.
- **Data leakage** (zbyt dobre wyniki na walidacji) i niewłaściwy podział (czasowy/grupowy).
- Spójność preprocessingu między treningiem a inferencją.
- Niezbalansowane klasy — czy metryka jest właściwa (accuracy vs PR-AUC/F1)?

#### 2. Baseline i sanity checks
- Porównaj z prostym baseline'em (klasa większościowa, regresja logistyczna).
- **Overfit na małym zbiorze** (kilkanaście próbek): jeśli model nie osiąga ~0 loss, jest błąd w kodzie/architekturze/lossie.
- Sprawdź loss początkowy (np. `ln(K)` dla K klas), kształty tensorów, brak `model.eval()`, kolejność normalizacji, błędne etykiety w dataloaderze.

#### 3. Diagnoza bias vs variance (krzywe uczenia)
| Objaw | Diagnoza | Działania |
|---|---|---|
| Wysoki błąd train i val | underfitting (high bias) | większy model, więcej cech, dłuższy trening, mniejsza regularyzacja |
| Niski train, wysoki val | overfitting (high variance) | więcej danych, augmentacja, regularyzacja, dropout, early stopping, prostszy model |
| Niski val, ale słaby w produkcji | rozjazd rozkładów | poprawa próbkowania, monitoring drift |

#### 4. Optymalizacja
- Learning rate (najczęstszy winowajca), scheduler, batch size, inicjalizacja, gradient clipping, NaN/Inf, vanishing/exploding gradients (monitoruj normy gradientów).

#### 5. Analiza błędów
- Rozbij metryki na segmenty (klasy, grupy, długości), przejrzyj najgorsze predykcje, macierz pomyłek; szukaj wzorców — często to bardziej wartościowe niż strojenie hiperparametrów.

#### 6. Weryfikacja
- Zmiana po zmianie, kontrola wersji eksperymentów (MLflow/W&B), ustalone seedy, testy jednostkowe dla danych i transformacji.

**Źródła:**
- [A Recipe for Training Neural Networks (Karpathy)](https://karpathy.github.io/2019/04/25/recipe/)
- [Machine Learning Yearning (Andrew Ng)](https://info.deeplearning.ai/machine-learning-yearning-book)
- [Rules of Machine Learning (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)

---

<a id="q280"></a>
### 280. Jak zapewnić sprawiedliwość (fairness) i ograniczyć bias w modelach ML?

**Odpowiedź:**

Bias w ML może pochodzić z danych (historyczne uprzedzenia, niereprezentatywna próba, błędy etykiet, proxy cech wrażliwych), z doboru celu/metryki oraz z samego sposobu użycia modelu. Nie da się go „usunąć" jednym trikiem; potrzebny jest proces na całym cyklu życia.

#### Definicje fairness (często wzajemnie sprzeczne)
- **Demographic parity**: `P(ŷ=1 | A=a)` jednakowe dla grup.
- **Equalized odds**: równe TPR i FPR w grupach; **equal opportunity**: równe TPR.
- **Calibration**: dla danego wyniku prawdopodobieństwo zdarzenia takie samo w grupach.
- Poza szczególnymi przypadkami nie można spełnić wszystkich naraz (twierdzenia o niemożliwości), więc wybór zależy od kontekstu i skutków.

#### Interwencje
- **Przed treningiem (pre-processing)**: zbieranie bardziej reprezentatywnych danych, reweighting, resampling, poprawa etykiet.
- **W trakcie (in-processing)**: regularyzacja/ograniczenia fairness, adversarial debiasing.
- **Po treningu (post-processing)**: dobór progów per grupa, kalibracja.
- Uwaga: samo usunięcie cechy wrażliwej nie wystarcza (proxy, np. kod pocztowy).

#### Proces
- Audyt: metryki rozbite na podgrupy (slicing), analiza błędów, testy porównawcze (Fairlearn, AIF360).
- Zaangażowanie interesariuszy, dokumentacja (**model cards**, datasheets for datasets).
- Monitoring po wdrożeniu, human-in-the-loop, mechanizmy odwołań.
- Zgodność z prawem (RODO, przepisy antydyskryminacyjne, EU AI Act).

**Źródła:**
- [A Survey on Bias and Fairness in Machine Learning](https://arxiv.org/abs/1908.09635)
- [Equality of Opportunity in Supervised Learning](https://arxiv.org/abs/1610.02413)
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)
- [Fairness and Machine Learning (Barocas, Hardt, Narayanan)](https://fairmlbook.org/)
- [Fairlearn — dokumentacja](https://fairlearn.org/)

---

<a id="q281"></a>
### 281. Wyjaśnij MLOps i jego kluczowe komponenty.

**Odpowiedź:**

**MLOps** to zestaw praktyk łączących ML, DevOps i data engineering, aby niezawodnie, powtarzalnie i szybko wdrażać oraz utrzymywać modele w produkcji. Rozszerza CI/CD o elementy specyficzne dla ML: dane i modele podlegają zmianom, więc oprócz kodu wersjonujemy dane, eksperymenty i modele, a system wymaga ciągłego trenowania (**CT**) i monitoringu.

#### Kluczowe komponenty
- **Wersjonowanie**: kod (Git), dane (DVC, lakeFS), modele, konfiguracje.
- **Data pipelines i walidacja danych**: ETL/ELT, orkiestracja (Airflow, Kubeflow, Dagster), walidacja schematu i rozkładów (Great Expectations, TFDV).
- **Feature store**: spójne cechy offline/online.
- **Experiment tracking**: MLflow, Weights & Biases — hiperparametry, metryki, artefakty.
- **Trening i pipeline'y ML**: reprodukowalne, zautomatyzowane, z retrainingiem wg harmonogramu lub triggerów (drift).
- **Model registry**: wersje, stadia (staging/production), metadane, zatwierdzanie.
- **CI/CD/CT**: testy kodu, danych i modelu; automatyczne wdrożenia (canary, blue-green, shadow).
- **Serwowanie**: kontenery, Kubernetes, serwery inferencji (Triton, KServe, TorchServe).
- **Monitoring i observability**: latency, błędy, data/concept drift, jakość predykcji, koszty; alerty.
- **Governance**: audyt, model cards, kontrola dostępu, zgodność.

#### Poziomy dojrzałości (wg Google)
0: proces manualny; 1: zautomatyzowany pipeline treningowy (CT); 2: pełne CI/CD pipeline'ów.

**Źródła:**
- [MLOps: Continuous delivery and automation pipelines (Google Cloud)](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [ml-ops.org — MLOps Principles](https://ml-ops.org/)
- [MLflow — dokumentacja](https://mlflow.org/docs/latest/index.html)
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html)

---

<a id="q282"></a>
### 282. Czym jest feature store i dlaczego jest ważny?

**Odpowiedź:**

**Feature store** to scentralizowana platforma do definiowania, obliczania, przechowywania, wersjonowania i serwowania cech (features) używanych w treningu i inferencji.

#### Architektura
- **Offline store** (hurtownia/data lake: BigQuery, S3+Parquet) — duże zbiory historyczne do treningu.
- **Online store** (Redis, DynamoDB, Cassandra) — niskolatencyjne pobieranie najnowszych wartości dla inferencji online.
- **Rejestr cech** (metadane, właściciele, definicje) i silnik transformacji (batch/streaming).
- Przykłady: Feast, Tecton, Vertex AI Feature Store, SageMaker Feature Store, Databricks Feature Store.

#### Dlaczego jest ważny
- **Eliminuje training-serving skew**: jedna definicja cechy dla treningu i produkcji.
- **Point-in-time correctness**: przy budowie zbioru treningowego pobiera wartości cech znane w momencie zdarzenia, unikając data leakage z przyszłości.
- **Reużywalność i odkrywalność**: zespoły współdzielą cechy zamiast implementować je od nowa.
- **Niska latencja** serwowania i pre-obliczanie kosztownych cech.
- **Governance**: lineage, wersjonowanie, kontrola dostępu, monitoring jakości cech.

#### Kiedy się przydaje, a kiedy nie
- Wiele modeli i zespołów współdzielących cechy, wymagania real-time — tak.
- Pojedynczy prosty model batch — narzut operacyjny może się nie opłacać.

```python
# Feast: pobranie cech online
features = store.get_online_features(
    features=["user_stats:avg_order_value", "user_stats:orders_30d"],
    entity_rows=[{"user_id": 1001}],
).to_dict()
```

**Źródła:**
- [Feast — dokumentacja](https://docs.feast.dev/)
- [feast.dev — What is a Feature Store?](https://feast.dev/blog/what-is-a-feature-store/)
- [Rules of Machine Learning (Google)](https://developers.google.com/machine-learning/guides/rules-of-ml)

---

<a id="q283"></a>
### 283. Wdrożenie modelu w chmurze vs on-device.

**Odpowiedź:**

Wybór miejsca uruchamiania inferencji to kompromis między mocą obliczeniową, latency, prywatnością, kosztem i dostępnością offline.

#### Chmura (cloud)
**Zalety:**
- Duża moc obliczeniowa (GPU/TPU), możliwość uruchamiania dużych modeli (LLM).
- Łatwe aktualizacje modelu bez udziału użytkownika, centralny monitoring i A/B testy.
- Elastyczne skalowanie.

**Wady:**
- Opóźnienie sieciowe, zależność od łączności.
- Koszt inferencji rośnie z ruchem.
- Dane użytkownika opuszczają urządzenie (prywatność, regulacje).

#### On-device (edge)
**Zalety:**
- Niska latency, działanie offline.
- Prywatność: dane zostają lokalnie (można łączyć z federated learning).
- Brak kosztów serwerowych na inferencję.

**Wady:**
- Ograniczone zasoby (pamięć, CPU/NPU, bateria, temperatura) — konieczna kompresja: quantization, pruning, distillation.
- Trudniejsze aktualizacje i fragmentacja urządzeń.
- Model narażony na ekstrakcję/inżynierię wsteczną.
- Trudniejszy monitoring jakości.

#### Narzędzia on-device
TensorFlow Lite, PyTorch ExecuTorch/Mobile, Core ML, ONNX Runtime Mobile, MediaPipe.

#### Podejścia hybrydowe
- Mały model lokalnie do szybkich/prywatnych zadań, cięższy w chmurze do trudnych (fallback/cascade).
- Split computing (część sieci na urządzeniu, część w chmurze).
- Przykład: wykrywanie słowa wybudzającego lokalnie, pełne rozpoznawanie mowy w chmurze.

**Źródła:**
- [Cloud vs On-Device Model Deployment (Outcome School)](https://outcomeschool.com/blog/cloud-vs-on-device-model-deployment)
- [TensorFlow Lite — dokumentacja](https://www.tensorflow.org/lite)
- [ExecuTorch — dokumentacja](https://pytorch.org/executorch/)
- [Communication-Efficient Learning of Deep Networks from Decentralized Data (Federated Learning)](https://arxiv.org/abs/1602.05629)

---

<a id="q284"></a>
### 284. Omów techniki kompresji modeli (Model Compression).

**Odpowiedź:**

Kompresja modeli zmniejsza rozmiar, zużycie pamięci, energii i latency, zwykle kosztem niewielkiej utraty jakości — kluczowa przy wdrożeniach on-device i przy kosztownych LLM.

#### 1. Quantization
Redukcja precyzji wag/aktywacji (FP32 -> FP16/BF16 -> INT8 -> INT4).
- **Post-training quantization (PTQ)**: bez ponownego treningu, szybka; wymaga kalibracji.
- **Quantization-aware training (QAT)**: symulacja kwantyzacji w treningu; lepsza jakość przy niskich bitach.
- INT8 zmniejsza rozmiar ok. 4x względem FP32. Dla LLM: GPTQ, AWQ, bitsandbytes.

#### 2. Pruning
Usuwanie mało istotnych wag lub struktur.
- **Unstructured** (pojedyncze wagi; rzadkie macierze — przyspieszenie wymaga wsparcia sprzętowego).
- **Structured** (neurony, kanały, głowice attention; realne przyspieszenie).
- Zwykle iteracyjnie: prune -> fine-tune. **Lottery Ticket Hypothesis** sugeruje istnienie małych, trenowalnych podsieci.

#### 3. Knowledge distillation
Mały „uczeń" uczy się naśladować rozkład wyjściowy (soft labels, z temperaturą T) dużego „nauczyciela":
`L = α·CE(y, p_s) + (1-α)·T²·KL(p_t^T || p_s^T)`.
Przykłady: DistilBERT.

#### 4. Low-rank factorization
Aproksymacja macierzy wag iloczynem `W ≈ AB` o niskim rzędzie (SVD, LoRA dla adaptacji).

#### 5. Efektywne architektury i inne
- MobileNet, EfficientNet, dzielenie wag (weight sharing), NAS pod ograniczenia sprzętowe.
- Kompilatory/optymalizacja grafu (TensorRT, ONNX).
- **Deep Compression**: pruning + kwantyzacja + kodowanie Huffmana.

#### Wybór i ewaluacja
Kombinacje technik często się sumują; zawsze mierz kompromis jakość vs rozmiar vs latency na docelowym sprzęcie.

**Źródła:**
- [Model Optimization (materiał z README)](https://www.linkedin.com/posts/pallavi-shekhar_ai-machinelearning-modeloptimization-activity-7438824172996247552-x7Wt)
- [Deep Compression (Han i in.)](https://arxiv.org/abs/1510.00149)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
- [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

---

## Prawdopodobieństwo i statystyka

<a id="q285"></a>
### 285. Wyjaśnij kompromis między obciążeniem a wariancją (Bias-Variance Tradeoff).

**Odpowiedź:**

Oczekiwany błąd predykcji modelu w punkcie `x` (dla straty kwadratowej) rozkłada się na trzy składniki:

`E[(y - f̂(x))²] = Bias[f̂(x)]² + Var[f̂(x)] + σ²`

- **Bias** — systematyczny błąd wynikający z uproszczonych założeń modelu (`E[f̂(x)] - f(x)`); wysoki bias = **underfitting**.
- **Variance** — wrażliwość modelu na konkretny zbiór treningowy; wysoka wariancja = **overfitting** (model uczy się szumu).
- **σ² (irreducible error)** — szum w danych, którego nie da się zredukować.

#### Intuicja
Zwiększanie złożoności modelu (stopień wielomianu, głębokość drzewa, liczba parametrów) zmniejsza bias, ale zwiększa wariancję. Optimum leży pomiędzy: minimum błędu na zbiorze walidacyjnym.

| Model | Bias | Variance |
|---|---|---|
| Regresja liniowa | wysoki | niski |
| Głębokie drzewo | niski | wysoki |
| k-NN, małe k | niski | wysoki |
| Random Forest | niski | obniżona przez bagging |

#### Jak sterować
- **Redukcja wariancji**: więcej danych, regularyzacja (L1/L2, dropout), early stopping, bagging/ensemble, prostszy model, redukcja liczby cech.
- **Redukcja biasu**: bogatszy model, więcej/lepsze cechy, boosting, mniej regularyzacji.
- **Diagnoza**: krzywe uczenia (train vs val error).

#### Uwaga: double descent
W nowoczesnych, silnie przeparametryzowanych modelach (sieci głębokie) klasyczna krzywa U nie zawsze się sprawdza — obserwuje się zjawisko *double descent*, gdzie błąd testowy ponownie spada po przekroczeniu progu interpolacji.

**Źródła:**
- [Deep Learning Book — Machine Learning Basics (rozdz. 5)](https://www.deeplearningbook.org/contents/ml.html)
- [Bias–variance tradeoff (Wikipedia)](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- [Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118)
- [Understanding the Bias-Variance Tradeoff (Scott Fortmann-Roe)](https://scott.fortmann-roe.com/docs/BiasVariance.html)

---

<a id="q286"></a>
### 286. Wyjaśnij różne rozkłady prawdopodobieństwa (normalny, dwumianowy, Poissona, jednostajny).

**Odpowiedź:**

Rozkłady dzielimy na **dyskretne** (opisane funkcją prawdopodobieństwa PMF) i **ciągłe** (gęstość PDF).

| Rozkład | Typ | Parametry | Zastosowanie | Średnia / Wariancja |
|---|---|---|---|---|
| Jednostajny `U(a,b)` | ciągły | `a, b` | brak wiedzy, generowanie losowe | `(a+b)/2` / `(b-a)²/12` |
| Normalny `N(μ,σ²)` | ciągły | `μ, σ²` | błędy pomiaru, CLT, inicjalizacja wag | `μ` / `σ²` |
| Dwumianowy `Bin(n,p)` | dyskretny | `n, p` | liczba sukcesów w `n` próbach | `np` / `np(1-p)` |
| Poissona `Pois(λ)` | dyskretny | `λ` | liczba zdarzeń w przedziale czasu | `λ` / `λ` |

#### Wzory
- Uniform: `f(x) = 1/(b-a)` dla `x ∈ [a,b]`.
- Normalny: `f(x) = (1/(σ√(2π))) · exp(-(x-μ)²/(2σ²))`.
- Dwumianowy: `P(X=k) = C(n,k) p^k (1-p)^(n-k)`.
- Poissona: `P(X=k) = λ^k e^(-λ) / k!`.

#### Związki między rozkładami
- Dla dużego `n` i małego `p` `Bin(n,p) ≈ Pois(np)`.
- Dla dużego `n` `Bin(n,p) ≈ N(np, np(1-p))` (twierdzenie de Moivre'a-Laplace'a).
- Suma wielu niezależnych zmiennych o skończonej wariancji dąży do normalnego (**CLT**).

#### W ML
Normalny: założenia regresji liniowej (szum gaussowski -> MSE), GMM, inicjalizacja, VAE. Dwumianowy/Bernoulliego: klasyfikacja binarna (log loss). Poisson: regresja Poissona dla zliczeń. Jednostajny: losowanie, prior nieinformatywny.

```python
from scipy import stats
stats.norm(0, 1).pdf(0); stats.binom(10, 0.5).pmf(5); stats.poisson(3).pmf(2)
```

**Źródła:**
- [scipy.stats — rozkłady](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Probability distribution (Wikipedia)](https://en.wikipedia.org/wiki/Probability_distribution)
- [Deep Learning Book — Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html)

---

<a id="q287"></a>
### 287. Czym jest rozkład normalny i jego funkcje?

**Odpowiedź:**

**Rozkład normalny (Gaussa)** `N(μ, σ²)` to ciągły rozkład symetryczny o kształcie dzwonu, w pełni opisany przez średnią `μ` (położenie) i wariancję `σ²` (rozrzut).

#### Funkcje
- **PDF**: `f(x) = 1/(σ√(2π)) · exp(-(x-μ)²/(2σ²))`.
- **CDF**: `F(x) = ½[1 + erf((x-μ)/(σ√2))]` (brak postaci elementarnej).
- **Standaryzacja**: `Z = (X-μ)/σ ~ N(0,1)`; wyniki z-score.
- Kwantyle, np. `z_{0.975} ≈ 1.96`.

#### Reguła 68-95-99,7
W przybliżeniu 68% masy leży w `μ ± σ`, 95% w `μ ± 2σ`, 99,7% w `μ ± 3σ`.

#### Własności
- Średnia = mediana = moda (symetria).
- Suma niezależnych normalnych jest normalna: `N(μ₁,σ₁²)+N(μ₂,σ₂²)=N(μ₁+μ₂,σ₁²+σ₂²)`.
- Przy zadanej średniej i wariancji ma **maksymalną entropię**.
- Centralne twierdzenie graniczne uzasadnia jej powszechność.
- Wielowymiarowy: `N(μ, Σ)` z macierzą kowariancji `Σ`.

#### Znaczenie w ML
- MLE dla szumu gaussowskiego prowadzi do **MSE**.
- Gaussian Naive Bayes, LDA/QDA, GMM, procesy gaussowskie.
- Inicjalizacje wag (Xavier/He), szum w VAE i modelach dyfuzyjnych.
- Testy statystyczne (t, z) i przedziały ufności opierają się na normalności lub CLT.
- Sprawdzanie normalności: QQ-plot, test Shapiro-Wilka; skośne dane można przekształcić (log, Box-Cox).

**Źródła:**
- [Normal distribution (Wikipedia)](https://en.wikipedia.org/wiki/Normal_distribution)
- [scipy.stats.norm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html)
- [Deep Learning Book — Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html)

---

<a id="q288"></a>
### 288. Czym jest rozkład wykładniczy?

**Odpowiedź:**

**Rozkład wykładniczy** `Exp(λ)` opisuje czas oczekiwania do pierwszego zdarzenia w procesie Poissona (zdarzenia zachodzą niezależnie ze stałą intensywnością `λ`).

#### Funkcje
- **PDF**: `f(x) = λ e^(-λx)`, `x ≥ 0`.
- **CDF**: `F(x) = 1 - e^(-λx)`.
- **Funkcja przeżycia**: `S(x) = P(X>x) = e^(-λx)`.
- **Średnia** `1/λ`, **wariancja** `1/λ²`, **mediana** `ln2/λ`.

#### Własność braku pamięci (memorylessness)
`P(X > s+t | X > s) = P(X > t)`. Jest to jedyny ciągły rozkład o tej własności (odpowiednik dyskretny: rozkład geometryczny). Przykład: jeśli zdarzenie nie nastąpiło do czasu `s`, dalszy czas oczekiwania ma ten sam rozkład.

#### Związki
- Czasy między zdarzeniami procesu Poissona `Pois(λ)` mają rozkład `Exp(λ)`.
- `Exp(λ)` to szczególny przypadek rozkładu gamma (`k=1`) i Weibulla (`k=1`).
- Suma `k` niezależnych `Exp(λ)` daje rozkład Erlanga/gamma.

#### Zastosowania
- Czas między przybyciami żądań (teoria kolejek), czas życia elementów bez zużycia, analiza przeżycia (stałe hazard rate).
- **MLE**: `λ̂ = 1/x̄`.
- W ML: modelowanie czasów zdarzeń, prior dla parametrów dodatnich, próbkowanie (inverse transform: `x = -ln(U)/λ`).

```python
import numpy as np
x = np.random.exponential(scale=1/2.0, size=10_000)   # lambda = 2
print(x.mean())  # ~0.5
```

**Źródła:**
- [Exponential distribution (Wikipedia)](https://en.wikipedia.org/wiki/Exponential_distribution)
- [scipy.stats.expon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html)
- [Poisson point process (Wikipedia)](https://en.wikipedia.org/wiki/Poisson_point_process)

---

<a id="q289"></a>
### 289. Czym jest rozkład dwumianowy (binomial)?

**Odpowiedź:**

**Rozkład dwumianowy** `Bin(n, p)` opisuje liczbę sukcesów `X` w `n` niezależnych próbach Bernoulliego, każda z prawdopodobieństwem sukcesu `p`.

#### Założenia
1. Stała liczba prób `n`.
2. Dwa możliwe wyniki w każdej próbie.
3. Stałe `p` w każdej próbie.
4. Niezależność prób.

#### Funkcje i momenty
- **PMF**: `P(X=k) = C(n,k) p^k (1-p)^(n-k)`, gdzie `C(n,k) = n!/(k!(n-k)!)`.
- **Średnia** `np`, **wariancja** `np(1-p)`.
- **CDF**: suma PMF dla `0..k`.

#### Przykład
Rzut monetą 10 razy (`p=0,5`): `P(X=5) = C(10,5)/2¹⁰ = 252/1024 ≈ 0,246`.

#### Przybliżenia
- Dla dużego `n` i małego `p`: Poisson z `λ=np`.
- Dla dużego `n` i `p` niezbyt bliskiego 0/1 (`np` i `n(1-p)` ≳ 10): normalny `N(np, np(1-p))`.

#### Zastosowania w ML
- Liczba poprawnych klasyfikacji w zbiorze testowym (przedział ufności dla accuracy); test dwumianowy.
- A/B testy konwersji (test dla proporcji).
- Model generatywny dla zmiennych binarnych; regresja logistyczna modeluje `p` dla rozkładu Bernoulliego/dwumianowego.
- Dropout: liczba aktywnych neuronów ~ dwumianowy.

```python
from scipy.stats import binom
binom.pmf(5, n=10, p=0.5)   # 0.246
binom.cdf(5, 10, 0.5)
```

**Źródła:**
- [Binomial distribution (Wikipedia)](https://en.wikipedia.org/wiki/Binomial_distribution)
- [scipy.stats.binom](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)
- [Deep Learning Book — Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html)

---

<a id="q290"></a>
### 290. Czym jest rozkład Bernoulliego?

**Odpowiedź:**

**Rozkład Bernoulliego** `Bernoulli(p)` opisuje pojedynczą próbę z dwoma wynikami: sukces (`1`) z prawdopodobieństwem `p` lub porażka (`0`) z `1-p`.

#### Funkcje i momenty
- **PMF**: `P(X=x) = p^x (1-p)^(1-x)`, `x ∈ {0,1}`.
- **Średnia** `p`, **wariancja** `p(1-p)` (maksymalna dla `p=0,5`).
- **Entropia**: `H = -p log p - (1-p) log(1-p)`.
- `Bin(n,p)` to suma `n` niezależnych zmiennych Bernoulliego.

#### Znaczenie w ML
- **Klasyfikacja binarna**: model zwraca `p = σ(z)`, a etykieta ~ `Bernoulli(p)`.
- **Log-likelihood** dla `N` próbek: `Σ [yᵢ ln pᵢ + (1-yᵢ) ln(1-pᵢ)]`; jego ujemna wartość to **binary cross-entropy** (log loss). Jest to zatem MLE dla modelu Bernoulliego.
- **MLE**: `p̂ = (liczba sukcesów)/N`.
- Naive Bayes Bernoulliego (cechy binarne, bag-of-words obecność/brak), RBM, dropout (maska Bernoulliego), Bernoulli VAE, bandyci wielorękowi (nagroda 0/1), Beta-Bernoulli w bayesowskim A/B testowaniu.

#### Uwaga
Rozkład kategorialny to uogólnienie Bernoulliego na `K > 2` wyników; multinomialny to uogólnienie dwumianowego.

```python
import torch.nn.functional as F
loss = F.binary_cross_entropy_with_logits(logits, y.float())
```

**Źródła:**
- [Bernoulli distribution (Wikipedia)](https://en.wikipedia.org/wiki/Bernoulli_distribution)
- [scipy.stats.bernoulli](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html)
- [Deep Learning Book — Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html)

---

<a id="q291"></a>
### 291. Czym jest rozkład wielomianowy (multinomial)?

**Odpowiedź:**

**Rozkład wielomianowy** `Multinomial(n, p₁,…,p_K)` uogólnia dwumianowy na `K` kategorii: opisuje wektor zliczeń `(X₁,…,X_K)` w `n` niezależnych próbach, gdzie w każdej próbie kategoria `i` pojawia się z prawdopodobieństwem `pᵢ` (`Σpᵢ = 1`).

#### Funkcje i momenty
- **PMF**: `P(x₁,…,x_K) = n!/(x₁!…x_K!) · ∏ pᵢ^{xᵢ}`, przy `Σxᵢ = n`.
- `E[Xᵢ] = npᵢ`, `Var[Xᵢ] = npᵢ(1-pᵢ)`, `Cov(Xᵢ,Xⱼ) = -npᵢpⱼ` (ujemna korelacja: suma jest stała).
- Dla `K=2` -> dwumianowy; dla `n=1` -> rozkład kategorialny.

#### Przykład
Rzut kostką 12 razy: prawdopodobieństwo, że każda ścianka wypadnie po 2 razy: `12!/(2!)⁶ · (1/6)¹² ≈ 0,0034`.

#### Zastosowania w ML
- **Multinomial Naive Bayes** dla tekstu (zliczenia słów, TF): `P(doc|c) ∝ ∏ p_{w|c}^{count_w}`.
- **Softmax + cross-entropy**: etykieta ~ kategorialny z prawdopodobieństwami z softmaxa.
- Modele tematyczne (LDA używa Dirichleta jako prior dla parametrów multinomialnych), próbkowanie tokenów w modelach językowych (`torch.multinomial`).
- **Prior sprzężony**: rozkład Dirichleta (Beta dla dwumianowego).
- **MLE**: `p̂ᵢ = xᵢ/n`; przy wygładzaniu Laplace'a (`α`) `p̂ᵢ = (xᵢ+α)/(n+Kα)`.

```python
import numpy as np
np.random.multinomial(n=12, pvals=[1/6]*6)
```

**Źródła:**
- [Multinomial distribution (Wikipedia)](https://en.wikipedia.org/wiki/Multinomial_distribution)
- [scikit-learn: Naive Bayes (MultinomialNB)](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [scipy.stats.multinomial](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html)

---

<a id="q292"></a>
### 292. Czym jest rozkład logarytmiczno-normalny (lognormal)?

**Odpowiedź:**

Zmienna `X` ma rozkład **lognormalny** `LogNormal(μ, σ²)`, jeśli `ln X ~ N(μ, σ²)`, czyli `X = e^Y` dla `Y` normalnego. Przyjmuje tylko wartości dodatnie i jest prawostronnie skośny.

#### Funkcje i momenty
- **PDF**: `f(x) = 1/(xσ√(2π)) · exp(-(ln x - μ)²/(2σ²))`, `x > 0`.
- **Średnia** `e^(μ+σ²/2)`, **mediana** `e^μ`, **moda** `e^(μ-σ²)` (średnia > mediana > moda).
- **Wariancja** `(e^(σ²) - 1) e^(2μ+σ²)`.
- Parametry `μ, σ` dotyczą logarytmu zmiennej, nie jej samej.

#### Skąd się bierze
Normalny wynika z **sumy** wielu niezależnych czynników; lognormalny z **iloczynu** wielu dodatnich niezależnych czynników (bo `ln` iloczynu to suma).

#### Zastosowania
- Dochody, ceny akcji (model Blacka-Scholesa), rozmiary plików, czasy odpowiedzi/latency, stężenia w biologii, długości tekstów.
- **Feature engineering**: dla skośnych cech dodatnich stosujemy transformację `log`/`log1p`, po której dane są zbliżone do normalnych — pomaga modelom liniowym, sieciom i redukuje wpływ outlierów.
- Testowanie: jeśli `log(x)` przechodzi test normalności, `x` jest w przybliżeniu lognormalne.
- **MLE**: `μ̂ = mean(ln xᵢ)`, `σ̂² = var(ln xᵢ)`.

```python
import numpy as np
y = np.log1p(df["income"])   # transformacja skośnej cechy
```

**Źródła:**
- [Log-normal distribution (Wikipedia)](https://en.wikipedia.org/wiki/Log-normal_distribution)
- [scipy.stats.lognorm](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html)
- [scikit-learn: Preprocessing — non-linear transformations](https://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation)

---

<a id="q293"></a>
### 293. Czym jest rozkład logistyczny?

**Odpowiedź:**

**Rozkład logistyczny** `Logistic(μ, s)` to ciągły, symetryczny rozkład o kształcie zbliżonym do normalnego, ale z **cięższymi ogonami**. Jego funkcja dystrybuanty to funkcja **sigmoidalna (logistyczna)**.

#### Funkcje
- **CDF**: `F(x) = 1/(1 + e^(-(x-μ)/s))`.
- **PDF**: `f(x) = e^(-(x-μ)/s) / (s(1 + e^(-(x-μ)/s))²)`.
- **Średnia** = mediana = moda = `μ`, **wariancja** `s²π²/3`.
- **Funkcja kwantylowa** (logit): `F⁻¹(p) = μ + s·ln(p/(1-p))`.

#### Związek z regresją logistyczną
Regresję logistyczną można wyprowadzić z modelu zmiennej ukrytej: `y* = wᵀx + ε`, gdzie `ε ~ Logistic(0,1)`, a `y = 1` gdy `y* > 0`. Wówczas
`P(y=1|x) = σ(wᵀx) = 1/(1+e^(-wᵀx))`.
Jeśli szum byłby normalny, otrzymamy **regresję probitową**. Współczynniki interpretujemy przez log-odds: `ln(p/(1-p)) = wᵀx`.

#### Inne zastosowania
- Funkcja aktywacji/sigmoid w sieciach neuronowych, LSTM (bramki).
- Model wzrostu logistycznego, model Elo/Bradleya-Terry'ego (prawdopodobieństwo zwycięstwa).
- Różnica dwóch zmiennych Gumbela ma rozkład logistyczny (powiązanie z softmaxem i Gumbel-max trick).

#### Różnica względem normalnego
Przy tej samej wariancji rozkład logistyczny ma wyższy szczyt i grubsze ogony, przez co jest bardziej odporny na obserwacje odstające w skrajnych obszarach.

**Źródła:**
- [Logistic distribution (Wikipedia)](https://en.wikipedia.org/wiki/Logistic_distribution)
- [Logistic regression (Wikipedia)](https://en.wikipedia.org/wiki/Logistic_regression)
- [scipy.stats.logistic](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.logistic.html)

---

<a id="q294"></a>
### 294. Czym jest rozkład gamma i jego funkcje?

**Odpowiedź:**

**Rozkład gamma** `Gamma(k, θ)` (kształt `k`, skala `θ`; alternatywnie `α, β=1/θ`) to ciągły rozkład na `x > 0`, opisujący m.in. czas oczekiwania na `k`-te zdarzenie w procesie Poissona.

#### Funkcje
- **PDF**: `f(x) = x^(k-1) e^(-x/θ) / (Γ(k) θ^k)`, gdzie `Γ(k) = ∫₀^∞ t^(k-1) e^(-t) dt` (dla całkowitych `Γ(k) = (k-1)!`).
- **Średnia** `kθ`, **wariancja** `kθ²`.
- **CDF**: niepełna funkcja gamma `γ(k, x/θ)/Γ(k)`.
- W parametryzacji `(α, β)` (rate): średnia `α/β`, wariancja `α/β²`.

#### Przypadki szczególne
- `k = 1` -> rozkład wykładniczy.
- `k` całkowite -> **rozkład Erlanga** (suma `k` niezależnych wykładniczych).
- `k = ν/2`, `θ = 2` -> **chi-kwadrat** z `ν` stopniami swobody.
- Odwrotny gamma i rozkład Beta pokrewne przez transformacje.

#### Zastosowania w ML i statystyce
- **Prior sprzężony** dla parametru `λ` rozkładu Poissona i dla precyzji (`1/σ²`) rozkładu normalnego w wnioskowaniu bayesowskim.
- Regresja gamma (GLM) dla dodatnich, skośnych celów, np. wartości szkód, czas obsługi.
- Modelowanie czasów oczekiwania, opadów, wielkości roszczeń.
- Rozkład Dirichleta można generować z niezależnych zmiennych gamma.

```python
from scipy.stats import gamma
gamma(a=2, scale=3).mean()   # k*theta = 6
```

**Źródła:**
- [Gamma distribution (Wikipedia)](https://en.wikipedia.org/wiki/Gamma_distribution)
- [scipy.stats.gamma](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html)
- [scikit-learn: GammaRegressor (GLM)](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-models)

---

<a id="q295"></a>
### 295. Rozkład Poissona i jego funkcja.

**Odpowiedź:**

**Rozkład Poissona** `Pois(λ)` modeluje liczbę zdarzeń zachodzących w ustalonym przedziale czasu/przestrzeni, gdy zdarzenia są niezależne i występują ze stałą średnią intensywnością `λ`.

#### Funkcje
- **PMF**: `P(X=k) = λ^k e^(-λ) / k!`, `k = 0,1,2,…`.
- **CDF**: `Σ_{i≤k} λ^i e^(-λ)/i!`.
- **Średnia = wariancja = λ** (własność „equidispersion" — jej naruszenie to over/under-dispersion).
- Suma niezależnych `Pois(λ₁)` i `Pois(λ₂)` to `Pois(λ₁+λ₂)`.

#### Przykład
Średnio 3 zgłoszenia na godzinę: `P(X=0) = e^(-3) ≈ 0,0498`; `P(X=2) = 9e^(-3)/2 ≈ 0,224`.

#### Założenia
Zdarzenia niezależne, stała intensywność, zdarzenia nie zachodzą dokładnie jednocześnie.

#### Zastosowania w ML
- **Regresja Poissona** (GLM z łączem log): `ln λ = wᵀx` — dane zliczeniowe (liczba kliknięć, awarii, zgłoszeń); w sklearn `PoissonRegressor`, w XGBoost `objective="count:poisson"`, w PyTorch `PoissonNLLLoss`.
- Strata: `λ - k ln λ + ln k!`.
- Modelowanie ruchu w systemach (kolejki), rzadkich zdarzeń, procesy punktowe (Poissona, Hawkesa).
- Gdy wariancja > średnia (overdispersion) — użyj ujemnego dwumianowego.

```python
from scipy.stats import poisson
poisson.pmf(2, mu=3)  # 0.224
```

**Źródła:**
- [Poisson distribution (Wikipedia)](https://en.wikipedia.org/wiki/Poisson_distribution)
- [scikit-learn: PoissonRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html)
- [scipy.stats.poisson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html)

---

<a id="q296"></a>
### 296. Kiedy użyć rozkładu Poissona zamiast dwumianowego?

**Odpowiedź:**

**Dwumianowy** modeluje liczbę sukcesów w **znanej, skończonej liczbie prób** `n` z prawdopodobieństwem sukcesu `p`. **Poissona** modeluje liczbę zdarzeń w przedziale czasu/przestrzeni, gdy **nie ma naturalnego górnego limitu** liczby prób i znamy tylko średnią intensywność `λ`.

#### Kiedy Poisson
- Nie znamy `n` ani `p` osobno, znamy tylko średnią liczbę zdarzeń (np. 4 awarie/miesiąc).
- Liczba możliwych „okazji" jest bardzo duża, a prawdopodobieństwo w każdej bardzo małe (rzadkie zdarzenia): `n` duże, `p` małe, `np = λ` umiarkowane.
- Zdarzenia zachodzą w sposób ciągły w czasie (połączenia telefoniczne, żądania HTTP, mutacje, kliknięcia).

#### Kiedy dwumianowy
- Jest stała liczba niezależnych prób z dwoma wynikami (10 rzutów monetą, 200 klientów, z których każdy kupuje lub nie).
- Wartość może być ograniczona przez `n` (nie może być więcej sukcesów niż prób).

#### Związek
`Bin(n, p) -> Pois(np)` gdy `n -> ∞`, `p -> 0`, `np = λ`. Praktyczna reguła: `n ≥ 20` i `p ≤ 0,05` (lub `n ≥ 100`, `np ≤ 10`).
- Dwumianowy: wariancja `np(1-p) < np` (under-dispersion względem Poissona); Poisson: wariancja = średnia.

#### Porównanie numeryczne
`n=1000, p=0,003` (`λ=3`): `P(X=2)` dwumianowy ≈ 0,2244, Poisson ≈ 0,2240 — niemal identyczne.

#### Praktyka
Do danych zliczeniowych bez ograniczenia górnego (regresja Poissona), do proporcji/sukcesów z ustalonej liczby prób (regresja logistyczna/dwumianowa). Gdy wariancja znacznie przekracza średnią — negative binomial.

**Źródła:**
- [Poisson distribution — Relation to binomial (Wikipedia)](https://en.wikipedia.org/wiki/Poisson_distribution)
- [Binomial distribution — Poisson approximation (Wikipedia)](https://en.wikipedia.org/wiki/Binomial_distribution)
- [scipy.stats.poisson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html)

---

<a id="q297"></a>
### 297. Czym jest wariancja?

**Odpowiedź:**

**Wariancja** mierzy rozrzut zmiennej losowej wokół jej wartości oczekiwanej: średni kwadrat odchylenia od średniej.

#### Wzory
- Populacja / zmienna losowa: `Var(X) = E[(X - μ)²] = E[X²] - (E[X])²`.
- Próbka (nieobciążony estymator): `s² = (1/(n-1)) Σ (xᵢ - x̄)²` — **korekta Bessela** (`n-1`), bo `x̄` jest szacowane z tych samych danych.
- NumPy: `np.var(x)` domyślnie `ddof=0` (dzieli przez `n`); pandas `Series.var()` domyślnie `ddof=1`.

#### Własności
- `Var(X) ≥ 0`; równa 0 tylko dla stałej.
- `Var(aX + b) = a² Var(X)`.
- Dla niezależnych: `Var(X+Y) = Var(X) + Var(Y)`; ogólnie `+ 2Cov(X,Y)`.
- Jednostka to kwadrat jednostki zmiennej (stąd użyteczność odchylenia standardowego).
- Wrażliwa na outliery (kwadrat).

#### Przykład
Dane `2, 4, 4, 4, 5, 5, 7, 9`: `x̄ = 5`, suma kwadratów odchyleń = 32; wariancja populacyjna `32/8 = 4`, próbkowa `32/7 ≈ 4,57`.

#### Znaczenie w ML
- **Bias-variance**: wariancja predykcji modelu względem zbioru treningowego.
- **PCA**: kierunki maksymalnej wariancji; **feature selection** (`VarianceThreshold`) — usuwanie cech o zerowej wariancji.
- Standaryzacja (dzielenie przez `σ`), Batch Normalization, inicjalizacja wag (Xavier/He utrzymują wariancję aktywacji).
- Redukcja wariancji gradientu (mini-batch, baseline w RL), bagging.

**Źródła:**
- [Variance (Wikipedia)](https://en.wikipedia.org/wiki/Variance)
- [Bessel's correction (Wikipedia)](https://en.wikipedia.org/wiki/Bessel%27s_correction)
- [numpy.var](https://numpy.org/doc/stable/reference/generated/numpy.var.html)

---

<a id="q298"></a>
### 298. Czym jest odchylenie standardowe (stddev)?

**Odpowiedź:**

**Odchylenie standardowe** to pierwiastek kwadratowy z wariancji: `σ = √Var(X)`. Mierzy typowy rozrzut wartości wokół średniej, **w tych samych jednostkach co dane** (w odróżnieniu od wariancji).

#### Wzory
- Populacja: `σ = √( (1/N) Σ (xᵢ - μ)² )`.
- Próbka: `s = √( (1/(n-1)) Σ (xᵢ - x̄)² )` (korekta Bessela; `s` jest wciąż lekko obciążonym estymatorem `σ`).

#### Interpretacja
- Małe `σ` -> dane skupione przy średniej; duże -> rozproszone.
- Dla rozkładu normalnego: ~68% w `μ ± σ`, ~95% w `μ ± 2σ`, ~99,7% w `μ ± 3σ`.
- **Nierówność Czebyszewa** (dowolny rozkład o skończonej wariancji): co najmniej `1 - 1/k²` masy w `μ ± kσ`.
- **Współczynnik zmienności** `CV = σ/μ` — porównanie rozrzutu skal.
- **Błąd standardowy średniej**: `SE = σ/√n` (to nie to samo co `σ`).

#### Przykład
Dane `2,4,4,4,5,5,7,9`: `σ = √4 = 2` (populacyjne).

#### Zastosowania w ML
- **Standaryzacja** (z-score): `z = (x - μ)/σ` — `StandardScaler`.
- Wykrywanie outlierów (`|z| > 3`), przedziały ufności, inicjalizacja wag, normalizacja (BatchNorm), niepewność predykcji (np. ensemble, MC dropout, GP).
- Rozrzut wyników walidacji krzyżowej (mean ± std) do porównania modeli.

```python
import numpy as np
x = np.array([2,4,4,4,5,5,7,9])
np.std(x), np.std(x, ddof=1)   # 2.0, 2.138
```

**Źródła:**
- [Standard deviation (Wikipedia)](https://en.wikipedia.org/wiki/Standard_deviation)
- [numpy.std](https://numpy.org/doc/stable/reference/generated/numpy.std.html)
- [scikit-learn: StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---

<a id="q299"></a>
### 299. Wyjaśnij różnicę między średnią, medianą i modą.

**Odpowiedź:**

To trzy miary tendencji centralnej.

| Miara | Definicja | Odporność na outliery | Typy danych |
|---|---|---|---|
| **Średnia** (mean) | suma / liczba obserwacji, `x̄ = (1/n)Σxᵢ` | niska | ilościowe |
| **Mediana** | wartość środkowa po posortowaniu (dla parzystej liczby: średnia dwóch środkowych) | wysoka | porządkowe, ilościowe |
| **Moda** | najczęstsza wartość | wysoka | także kategoryczne |

#### Przykład
Dane: `1, 2, 2, 3, 100`: średnia = 21,6; mediana = 2; moda = 2. Jeden outlier zawyżył średnią, mediana i moda pozostały reprezentatywne. Dlatego np. zarobki lub ceny domów raportuje się medianą.

#### Kształt rozkładu
- Symetryczny (np. normalny): średnia ≈ mediana ≈ moda.
- Prawostronnie skośny: średnia > mediana > moda (zwykle).
- Lewostronnie skośny: średnia < mediana < moda (zwykle).
- Rozkład może mieć wiele mód (bimodalny) albo nie mieć modalnej wartości.

#### Zastosowania w ML
- **Imputacja braków**: średnia (dane symetryczne), mediana (skośne/outliery), moda (kategoryczne) — `SimpleImputer(strategy=...)`.
- **Funkcje straty**: MSE minimalizowana przez średnią, MAE przez medianę (stąd większa odporność MAE).
- Klasyfikator bazowy (`DummyClassifier`: moda; `DummyRegressor`: średnia/mediana).
- **Głosowanie większościowe** w ensemble to moda; uśrednianie predykcji to średnia.
- Odporne skalowanie: `RobustScaler` (mediana i IQR).

```python
import numpy as np, scipy.stats as st
x = [1,2,2,3,100]
np.mean(x), np.median(x), st.mode(x).mode
```

**Źródła:**
- [Average / Central tendency (Wikipedia)](https://en.wikipedia.org/wiki/Central_tendency)
- [scikit-learn: SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)
- [scikit-learn: RobustScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)

---

<a id="q300"></a>
### 300. Jaka jest różnica między korelacją a kowariancją?

**Odpowiedź:**

Obie miary opisują **liniową współzmienność** dwóch zmiennych, ale różnią się skalą i interpretacją.

#### Kowariancja
`Cov(X,Y) = E[(X-μ_X)(Y-μ_Y)] = E[XY] - E[X]E[Y]`
Estymator z próbki: `(1/(n-1)) Σ (xᵢ - x̄)(yᵢ - ȳ)`.
- Znak mówi o kierunku (dodatnia: rosną razem; ujemna: przeciwnie).
- **Wartość zależy od jednostek/skali** (np. cm vs m zmienia kowariancję), więc nie da się porównywać siły zależności.
- `Cov(X,X) = Var(X)`.

#### Korelacja (Pearsona)
`ρ = Cov(X,Y)/(σ_X σ_Y)` — kowariancja znormalizowana, **bezwymiarowa**, w przedziale `[-1, 1]`, niezmiennicza względem skalowania i przesunięcia (dla dodatnich skal).

| | Kowariancja | Korelacja |
|---|---|---|
| Zakres | `(-∞, ∞)` | `[-1, 1]` |
| Jednostki | iloczyn jednostek | brak |
| Porównywalność | nie | tak |
| Zastosowanie | macierz kowariancji (PCA, GMM, LDA) | selekcja cech, EDA |

#### Uwagi
- Obie wykrywają tylko zależności **liniowe**; nieliniowa może dawać ~0 (np. `Y = X²` dla symetrycznego `X`).
- Niezależność -> zerowa kowariancja, ale nie odwrotnie (wyjątek: łączny rozkład normalny).
- Korelacja Pearsona wrażliwa na outliery; alternatywy: **Spearman**, **Kendall** (rangowe), mutual information.
- Macierz korelacji = macierz kowariancji zestandaryzowanych zmiennych.

```python
import numpy as np
np.cov(x, y)[0,1]; np.corrcoef(x, y)[0,1]
```

**Źródła:**
- [Covariance (Wikipedia)](https://en.wikipedia.org/wiki/Covariance)
- [Pearson correlation coefficient (Wikipedia)](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- [numpy.cov](https://numpy.org/doc/stable/reference/generated/numpy.cov.html)

---

<a id="q301"></a>
### 301. Co oznacza współczynnik korelacji +1, 0 i -1?

**Odpowiedź:**

Współczynnik korelacji Pearsona `r ∈ [-1, 1]` mierzy siłę i kierunek **liniowej** zależności.

- **`r = +1`**: idealna dodatnia zależność liniowa — wszystkie punkty leżą na prostej o dodatnim nachyleniu (`y = ax + b`, `a > 0`).
- **`r = -1`**: idealna ujemna zależność liniowa (prosta o ujemnym nachyleniu).
- **`r = 0`**: brak zależności **liniowej**. Nie oznacza braku jakiejkolwiek zależności ani niezależności — np. `Y = X²` przy symetrycznym `X` daje `r ≈ 0`, mimo że `Y` jest funkcją `X`.

#### Wartości pośrednie
Orientacyjnie (zależnie od dziedziny): `|r| < 0,3` słaba, `0,3-0,7` umiarkowana, `> 0,7` silna. `r²` to odsetek wariancji `Y` wyjaśniony liniowo przez `X` (w regresji prostej to współczynnik determinacji `R²`).

#### Pułapki
- **Outliery** mogą sztucznie zawyżyć lub zbić `r`.
- **Anscombe's quartet**: cztery zbiory o tej samej `r ≈ 0,82`, lecz zupełnie różnym kształcie — zawsze rób wykres.
- Ograniczony zakres zmiennej osłabia korelację.
- **Korelacja ≠ przyczynowość**.
- Dla zależności monotonicznych, nieliniowych — Spearman/Kendall.

#### W ML
- Wykrywanie **multikolinearności** (`|r|` blisko 1 między cechami -> niestabilne współczynniki; usuń lub użyj regularyzacji/PCA).
- Cechy silnie skorelowane z celem są kandydatami, ale nieliniowe zależności wymagają MI lub modeli drzewiastych.
- Korelacja predykcji modeli w ensemble (niska korelacja -> większy zysk).

```python
df.corr(method="pearson")   # lub "spearman"
```

**Źródła:**
- [Pearson correlation coefficient (Wikipedia)](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
- [Anscombe's quartet (Wikipedia)](https://en.wikipedia.org/wiki/Anscombe%27s_quartet)
- [pandas.DataFrame.corr](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)

---

<a id="q302"></a>
### 302. Wyjaśnij korelację a przyczynowość (Correlation vs Causation).

**Odpowiedź:**

**Korelacja** oznacza statystyczny związek między zmiennymi; **przyczynowość** — że zmiana jednej zmiennej *powoduje* zmianę drugiej. Z korelacji `X` i `Y` nie wynika, że `X` powoduje `Y`.

#### Możliwe wyjaśnienia korelacji X~Y
1. `X -> Y` (przyczynowość).
2. `Y -> X` (odwrócona przyczynowość).
3. **Zmienna zakłócająca (confounder)** `Z -> X` i `Z -> Y` — klasyk: sprzedaż lodów i utonięcia korelują (temperatura/lato).
4. **Selection bias / collider bias** (warunkowanie na wspólnym skutku).
5. Przypadek (spurious correlation), zwłaszcza przy wielu porównaniach lub krótkich szeregach czasowych.
6. **Paradoks Simpsona**: trend odwraca się po podziale na podgrupy.

#### Jak ustalać przyczynowość
- **Eksperymenty randomizowane (A/B testy, RCT)** — randomizacja neutralizuje confoundery; złoty standard.
- **Metody quasi-eksperymentalne**: difference-in-differences, regression discontinuity, zmienne instrumentalne, propensity score matching, synthetic control.
- **Causal inference**: grafy przyczynowe (DAG), do-calculus (Pearl), analiza kontrfaktyczna; biblioteki DoWhy, EconML.

#### Znaczenie w ML
- Modele predykcyjne uczą się korelacji; działają dobrze w rozkładzie treningowym, ale mogą zawodzić przy interwencji lub zmianie rozkładu (np. model uczy się „szpital = wyższa śmiertelność").
- Feature importance i SHAP **nie** oznaczają przyczynowości.
- Decyzje o interwencji (np. czy kampania zwiększa sprzedaż) wymagają uplift modeling/causal ML, nie samej predykcji.

**Źródła:**
- [Correlation does not imply causation (Wikipedia)](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation)
- [Simpson's paradox (Wikipedia)](https://en.wikipedia.org/wiki/Simpson%27s_paradox)
- [DoWhy — dokumentacja](https://www.pywhy.org/dowhy/)
- [Causal inference: What If (Hernán, Robins)](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)

---

<a id="q303"></a>
### 303. Czym są błędy typu I i typu II?

**Odpowiedź:**

W testowaniu hipotez rozważamy hipotezę zerową `H₀` (np. brak efektu) i alternatywną `H₁`.

| | `H₀` prawdziwa | `H₀` fałszywa |
|---|---|---|
| Odrzucamy `H₀` | **Błąd I rodzaju** (fałszywy alarm, false positive), prawdopodobieństwo `α` | poprawna decyzja (moc `1-β`) |
| Nie odrzucamy `H₀` | poprawna decyzja | **Błąd II rodzaju** (false negative), prawdopodobieństwo `β` |

- **Błąd I rodzaju**: uznajemy efekt za istniejący, choć go nie ma. `α` to poziom istotności (zwykle 0,05).
- **Błąd II rodzaju**: nie wykrywamy istniejącego efektu. `β = 1 - power`; **moc testu** (power) zwykle celuje się na 0,8.

#### Zależności
- Zmniejszenie `α` zwiększa `β` (przy stałej próbie).
- Moc rośnie wraz z **rozmiarem próby**, **wielkością efektu** i `α`, a maleje z szumem (wariancją).
- Analiza mocy przed eksperymentem określa wymaganą liczebność próby.

#### Analogia z ML
- Błąd I = **False Positive** (FPR = `α`); błąd II = **False Negative** (FNR = `β`); moc = **recall/TPR**.
- Wybór progu klasyfikatora to kompromis między nimi. Koszt zależy od kontekstu: przy diagnostyce raka ważniejszy jest niski FN (wysoki recall), przy filtrze spamu niski FP.

#### Testy wielokrotne
Przy wielu testach rośnie łączne ryzyko błędu I rodzaju (FWER) — stosuje się poprawki Bonferroniego lub kontrolę FDR (Benjamini-Hochberg). Dotyczy to też porównywania wielu modeli/hiperparametrów.

```python
from statsmodels.stats.power import TTestIndPower
TTestIndPower().solve_power(effect_size=0.3, alpha=0.05, power=0.8)  # ~175 na grupę
```

**Źródła:**
- [Type I and type II errors (Wikipedia)](https://en.wikipedia.org/wiki/Type_I_and_type_II_errors)
- [Statistical power (Wikipedia)](https://en.wikipedia.org/wiki/Power_of_a_test)
- [statsmodels — power analysis](https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations)

---

<a id="q304"></a>
### 304. Czym jest p-value? Co oznacza istotność statystyczna?

**Odpowiedź:**

**p-value** to prawdopodobieństwo uzyskania wyniku **co najmniej tak skrajnego** jak zaobserwowany, **przy założeniu, że hipoteza zerowa `H₀` jest prawdziwa** (i spełnione są założenia modelu testu):

`p = P(T ≥ t_obs | H₀)` (dla testu jednostronnego).

#### Procedura
1. Sformułuj `H₀` i `H₁`.
2. Wybierz statystykę testową i poziom istotności `α` (zwykle 0,05) *przed* analizą.
3. Oblicz p-value.
4. Jeśli `p ≤ α`, odrzucamy `H₀` — wynik jest **statystycznie istotny**; inaczej nie mamy podstaw do jej odrzucenia (to nie dowodzi, że `H₀` jest prawdziwa).

#### Przykład
Testujemy, czy nowa wersja strony zmienia konwersję. `p = 0,03 < 0,05`: jeśli różnicy naprawdę nie ma, taką lub większą różnicę zaobserwowalibyśmy w ok. 3% eksperymentów.

#### Czym p-value NIE jest
- Nie jest prawdopodobieństwem, że `H₀` jest prawdziwa.
- Nie jest prawdopodobieństwem, że wynik wynika z przypadku.
- Nie mierzy wielkości ani ważności efektu.
- `p > 0,05` nie oznacza „brak efektu".

#### Istotność statystyczna a praktyczna
Przy dużej próbie nawet znikomy efekt daje bardzo małe `p`; zawsze raportuj **wielkość efektu** (Cohen's d, różnica średnich) i **przedział ufności**.

```python
from scipy import stats
t, p = stats.ttest_ind(a, b, equal_var=False)   # test Welcha
```

**Źródła:**
- [p-value (Wikipedia)](https://en.wikipedia.org/wiki/P-value)
- [The ASA Statement on p-Values: Context, Process, and Purpose](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108)
- [scipy.stats.ttest_ind](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)

---

<a id="q305"></a>
### 305. Wyjaśnij p-value i jego ograniczenia.

**Odpowiedź:**

p-value to miara niezgodności danych z hipotezą zerową (`P(dane co najmniej tak skrajne | H₀)`). Jest użyteczne, ale często nadużywane.

#### Główne ograniczenia
1. **Nie mówi o prawdopodobieństwie hipotezy**: `P(dane|H₀) ≠ P(H₀|dane)`. Do tego potrzebne jest podejście bayesowskie (priory).
2. **Nie mierzy wielkości efektu**: przy ogromnej próbie trywialny efekt jest „istotny"; przy małej próbie ważny efekt bywa nieistotny.
3. **Zależy od rozmiaru próby** i wariancji.
4. **Arbitralny próg** 0,05 — `p = 0,049` i `0,051` niewiele się różnią, a decyzje binarne są mylące.
5. **p-hacking / data dredging**: testowanie wielu hipotez, podziałów lub specyfikacji aż do „istotnego" wyniku. Przy `m` niezależnych testach na poziomie 0,05 szansa choć jednego fałszywego pozytywu to `1-(0,95)^m` (dla `m=20` ok. 64%).
6. **Kryzys replikacji** i publication bias — istotne wyniki częściej się publikuje.
7. **Założenia testu** (niezależność, rozkład, jednorodność wariancji) — ich złamanie unieważnia p-value.
8. **Optional stopping**: zaglądanie do wyników A/B testu i zatrzymywanie przy `p<0,05` zawyża błąd I rodzaju; potrzebne testy sekwencyjne.
9. Brak informacji o precyzji: użyj przedziałów ufności.

#### Dobre praktyki
- Raportuj wielkość efektu + przedział ufności, nie tylko p.
- Zaplanuj analizę wcześniej (preregistration), przeprowadź analizę mocy.
- Korekty na testy wielokrotne (Bonferroni, Holm, Benjamini-Hochberg).
- Rozważ podejście bayesowskie (Bayes factor, posterior) lub metody resamplingowe (bootstrap, testy permutacyjne).
- Waliduj na niezależnych danych.

**Źródła:**
- [The ASA Statement on p-Values](https://www.tandfonline.com/doi/full/10.1080/00031305.2016.1154108)
- [Why Most Published Research Findings Are False (Ioannidis)](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.0020124)
- [Misuse of p-values (Wikipedia)](https://en.wikipedia.org/wiki/Misuse_of_p-values)
- [Multiple comparisons problem (Wikipedia)](https://en.wikipedia.org/wiki/Multiple_comparisons_problem)

---

<a id="q306"></a>
### 306. Czym jest testowanie hipotez i kiedy jest używane w ML?

**Odpowiedź:**

**Testowanie hipotez** to formalna procedura statystyczna decydowania, czy dane dostarczają wystarczających dowodów przeciwko hipotezie zerowej `H₀` na rzecz alternatywnej `H₁`, przy kontrolowanym ryzyku błędu I rodzaju `α`.

#### Kroki
1. Sformułuj `H₀` (np. `μ_A = μ_B`) i `H₁` (`μ_A ≠ μ_B` lub jednostronna).
2. Wybierz test i `α` (zwykle 0,05); zaplanuj wielkość próby (analiza mocy).
3. Oblicz statystykę testową i p-value.
4. Podejmij decyzję: odrzuć / nie odrzucaj `H₀`; raportuj efekt i przedział ufności.

#### Popularne testy
| Test | Zastosowanie |
|---|---|
| t-test (Studenta, Welcha, sparowany) | porównanie średnich |
| z-test dla proporcji | porównanie konwersji |
| chi-kwadrat | niezależność cech kategorycznych, zgodność rozkładów |
| ANOVA | średnie w wielu grupach |
| Mann-Whitney, Wilcoxon | odpowiedniki nieparametryczne |
| Kolmogorov-Smirnov | porównanie rozkładów (drift) |
| Shapiro-Wilk | normalność |
| McNemar | porównanie dwóch klasyfikatorów na tych samych danych |

#### Zastosowania w ML
- **A/B testy** modeli i funkcji produktowych.
- **Porównanie modeli**: sparowany t-test na foldach CV (z korektą, bo foldy nie są niezależne), test McNemara, testy permutacyjne/bootstrap.
- **Wykrywanie driftu** danych (KS, chi-kwadrat, PSI).
- **Selekcja cech** (test chi-kwadrat, ANOVA F-test: `SelectKBest`).
- Sprawdzanie założeń modeli (normalność reszt, heteroskedastyczność).
- Ocena istotności wyników (czy różnica w metryce to szum).

#### Pułapki
Testy wielokrotne, `p`-hacking, brak korekty, zależność próbek (CV), pomijanie wielkości efektu.

```python
from scipy import stats
stat, p = stats.ks_2samp(train_feature, prod_feature)   # drift cechy
```

**Źródła:**
- [Statistical hypothesis testing (Wikipedia)](https://en.wikipedia.org/wiki/Statistical_hypothesis_test)
- [scipy.stats — testy statystyczne](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms (Dietterich, 1998)](https://doi.org/10.1162/089976698300017197)
- [scikit-learn: feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)

---

<a id="q307"></a>
### 307. Jakich testów statystycznych użyjesz do porównania dwóch modeli?

**Odpowiedź:**

Wybór testu zależy od tego, **co porównujemy** (predykcje na tym samym zbiorze testowym, wyniki z cross-validation, wiele zbiorów danych) oraz od typu metryki. Kluczowe: oba modele oceniamy na **tych samych** danych, więc próbki są **sparowane**.

#### Klasyfikacja - ten sam zbiór testowy
- **Test McNemara** - porównuje dwa klasyfikatory na podstawie tabeli niezgodności: $b$ = przypadki, gdzie tylko model A jest poprawny, $c$ = tylko model B. Statystyka $\chi^2 = \frac{(|b-c|-1)^2}{b+c}$ (z poprawką na ciągłość), 1 stopień swobody. Tani (jeden zbiór testowy, brak retreningu), dobry dla dużych zbiorów testowych. Nie uwzględnia wariancji wynikającej z losowości treningu.
- **Bootstrap / test permutacyjny** na różnicy metryki (np. AUC, F1) - działa dla dowolnej metryki; próbkujemy przykłady testowe ze zwracaniem i budujemy przedział ufności dla $\Delta$ metryki. Dla AUC klasycznie także test DeLonga.

#### Wyniki z k-fold cross-validation
- **Sparowany test t-Studenta** na różnicach wyników z foldów. Uwaga: foldy treningowe się nakładają, więc różnice nie są niezależne, a test jest zbyt optymistyczny (zawyżone false positives).
- **Skorygowany (corrected resampled) t-test** Nadeau-Bengio lub **5x2 cv F-test** (Dietterich, Alpaydin) - poprawki na zależność między foldami.
- **Test Wilcoxona (signed-rank)** - nieparametryczna alternatywa, gdy różnice nie są normalne.

#### Wiele modeli / wiele zbiorów danych
- **Test Friedmana** + post-hoc **Nemenyi** (Demšar 2006) - rankingowe porównanie wielu algorytmów na wielu zbiorach; przy porównaniu parami - Wilcoxon z korektą Holma/Bonferroniego.

#### Praktyka
```python
from scipy import stats
import numpy as np
# wyniki A i B na tych samych 10 foldach
a = np.array([.81,.79,.83,.80,.82,.78,.84,.81,.80,.82])
b = np.array([.80,.78,.82,.80,.80,.77,.83,.80,.79,.81])
print(stats.ttest_rel(a, b))
print(stats.wilcoxon(a, b))
```
- Zawsze raportuj **wielkość efektu** i przedział ufności, nie tylko p-value; istotność statystyczna != istotność praktyczna.
- Przy wielu porównaniach stosuj korektę na wielokrotne testowanie.
- Ustal seedy i zbiór testowy, powtarzaj trening z różnymi seedami, aby uwzględnić wariancję inicjalizacji.
- W produkcji uzupełnij to testem A/B (np. test dla dwóch proporcji lub t-test na metryce biznesowej).

**Źródła:**
- [McNemar's test (Wikipedia)](https://en.wikipedia.org/wiki/McNemar%27s_test)
- [Demšar (2006): Statistical Comparisons of Classifiers over Multiple Data Sets (JMLR)](https://jmlr.org/papers/v7/demsar06a.html)
- [SciPy - statistical functions (ttest_rel, wilcoxon)](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Wilcoxon signed-rank test (Wikipedia)](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)

---

<a id="q308"></a>
### 308. Jak ocenić, czy cecha (feature) jest istotna statystycznie?

**Odpowiedź:**

Istotność statystyczna cechy oznacza, że jej związek ze zmienną celu jest mało prawdopodobny jako dzieło przypadku. Formalnie testujemy hipotezę zerową $H_0$: brak efektu (np. współczynnik $\beta_j = 0$) i liczymy **p-value** - prawdopodobieństwo uzyskania statystyki co najmniej tak ekstremalnej przy prawdziwym $H_0$. Jeśli $p < \alpha$ (zwykle 0,05), odrzucamy $H_0$.

#### Metody
- **Regresja liniowa (OLS)**: test t dla współczynnika, $t = \hat\beta_j / SE(\hat\beta_j)$; test F dla całego modelu lub grupy cech; przedziały ufności dla $\beta_j$.
- **Regresja logistyczna**: test Walda ($z = \hat\beta/SE$) lub **test ilorazu wiarygodności** (porównanie modeli z cechą i bez: $-2(\ell_0 - \ell_1) \sim \chi^2_k$).
- **Testy jednowymiarowe**: t-test/ANOVA (cecha numeryczna vs klasa), $\chi^2$ (kategoryczna vs kategoryczna), korelacja Pearsona/Spearmana, informacja wzajemna (mutual information).
- **Test permutacyjny**: tasujemy kolumnę cechy, mierzymy spadek jakości modelu; rozkład spadków daje p-value bez założeń rozkładowych (permutation importance, `sklearn.inspection.permutation_importance`).
- **Bootstrap** współczynników / ważności cech - przedział ufności.

#### Pułapki
- **Wielokrotne testowanie**: przy 1000 cech ok. 50 wyjdzie "istotnych" przy $\alpha=0{,}05$ przypadkiem - stosuj Bonferroniego, Holma lub kontrolę FDR (Benjamini-Hochberg).
- **Współliniowość** zawyża błędy standardowe (sprawdź VIF).
- Istotność != siła efektu ani ważność predykcyjna; przy dużym $n$ trywialne efekty są istotne. Raportuj wielkość efektu.
- Selekcja cech na całym zbiorze przed cross-validation powoduje data leakage.
- Zależność przyczynowa wymaga osobnej analizy.

```python
import statsmodels.api as sm
X = sm.add_constant(df[["x1","x2","x3"]])
res = sm.OLS(y, X).fit()
print(res.summary())   # t, p-value, przedziały ufności
```

**Źródła:**
- [Statistical significance (Wikipedia)](https://en.wikipedia.org/wiki/Statistical_significance)
- [p-value (Wikipedia)](https://en.wikipedia.org/wiki/P-value)
- [scikit-learn - Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html)
- [scikit-learn - Feature selection](https://scikit-learn.org/stable/modules/feature_selection.html)

---

<a id="q309"></a>
### 309. Czym jest przedział ufności (confidence interval) i jak się go stosuje?

**Odpowiedź:**

**Przedział ufności** to zakres wartości $[L, U]$ obliczony z próby, który przy wielokrotnym powtarzaniu eksperymentu zawierałby prawdziwy parametr populacji w ustalonym odsetku przypadków (poziom ufności, np. 95%). Interpretacja częstościowa: to **procedura** ma 95% pokrycia; nie jest to "95% prawdopodobieństwa, że parametr leży w tym konkretnym przedziale" (parametr jest stałą, losowy jest przedział).

#### Wzory
Dla średniej z dużej próby lub znanego $\sigma$:
$$\bar x \pm z_{1-\alpha/2}\frac{\sigma}{\sqrt n}$$
Dla nieznanego $\sigma$ i małej próby: $\bar x \pm t_{n-1,1-\alpha/2}\frac{s}{\sqrt n}$.
Dla proporcji (np. accuracy): $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ (Wald; lepiej przedział Wilsona przy małym $n$ lub $\hat p$ bliskim 0/1).

#### Zastosowania w ML
- **Niepewność metryki**: accuracy 0,90 na 100 przykładach ma przedział ok. $\pm 0{,}059$ (95%) - różnica 0,90 vs 0,92 to szum. Na 10 000 przykładach: $\pm 0{,}006$.
- **Porównanie modeli**: przedział dla różnicy metryk; jeśli zawiera 0, brak dowodu różnicy.
- **A/B testy**: przedział dla uplift.
- **Bootstrap CI** dla dowolnej metryki (AUC, F1).
- **Regresja**: przedziały dla współczynników, oraz przedziały predykcji (szersze niż przedziały ufności średniej).

```python
import numpy as np
rng = np.random.default_rng(0)
correct = rng.random(500) < 0.9          # wynik 0/1 dla każdego przykładu
boots = [rng.choice(correct, correct.size).mean() for _ in range(5000)]
print(np.percentile(boots, [2.5, 97.5]))
```

#### Wskazówki
- Wyższy poziom ufności = szerszy przedział; większe $n$ = węższy (skaluje się jak $1/\sqrt n$).
- Nie mylić z **przedziałem wiarygodności** (credible interval) w podejściu bayesowskim.
- Przedziały ufności zakładają niezależne próbki; przy skorelowanych danych (np. wiele wierszy tego samego użytkownika) są zbyt wąskie.

**Źródła:**
- [Confidence interval (Wikipedia)](https://en.wikipedia.org/wiki/Confidence_interval)
- [Binomial proportion confidence interval (Wikipedia)](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
- [Bootstrapping (Wikipedia)](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))

---

<a id="q310"></a>
### 310. Czym są z-score i t-score? Kiedy stosuje się każdy z nich?

**Odpowiedź:**

#### Z-score (wynik standaryzowany)
$$z = \frac{x-\mu}{\sigma}$$
Mówi, o ile odchyleń standardowych wartość $x$ leży od średniej. W ML używany do **standaryzacji cech** (`StandardScaler`, średnia 0, odchylenie 1), wykrywania outlierów (np. $|z|>3$) oraz w **teście z** dla średniej:
$$z = \frac{\bar x-\mu_0}{\sigma/\sqrt n}$$
Statystyka ma rozkład $N(0,1)$, gdy znamy prawdziwe $\sigma$ populacji lub próba jest duża.

#### T-score (statystyka t)
$$t = \frac{\bar x-\mu_0}{s/\sqrt n}$$
gdzie $s$ to odchylenie z próby. Statystyka ma rozkład **t-Studenta** z $n-1$ stopniami swobody - z cięższymi ogonami niż normalny, co uwzględnia dodatkową niepewność szacowania $\sigma$. Wraz ze wzrostem $n$ rozkład t zbiega do normalnego (dla $n>30$ różnice są małe).

#### Kiedy który
| Sytuacja | Test |
|---|---|
| $\sigma$ znane, rozkład normalny lub duże $n$ | z |
| $\sigma$ nieznane, małe $n$, dane w przybliżeniu normalne | t |
| Dwie średnie, niezależne próby | t dwóch prób (Welch, gdy wariancje różne) |
| Te same obiekty (np. dwa modele na tych samych foldach) | sparowany t |
| Proporcje (duże $n$) | test z dla proporcji |

W praktyce $\sigma$ prawie nigdy nie jest znane, więc **t-test jest domyślnym wyborem**. Uwaga: w psychometrii "T-score" bywa też inną skalą standaryzowaną ($50+10z$) - w statystyce testów chodzi o statystykę t.

```python
from scipy import stats
z = stats.zscore(x)                       # standaryzacja
t, p = stats.ttest_1samp(x, popmean=0)    # test t
```

Założenia: niezależność obserwacji i (dla małych prób) w przybliżeniu normalny rozkład; przy naruszeniu użyj testów nieparametrycznych lub bootstrapu.

**Źródła:**
- [Standard score (Wikipedia)](https://en.wikipedia.org/wiki/Standard_score)
- [Student's t-test (Wikipedia)](https://en.wikipedia.org/wiki/Student%27s_t-test)
- [scikit-learn - StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---

<a id="q311"></a>
### 311. Wyjaśnij twierdzenie Bayesa. Jak odnosi się do Naive Bayes i metod bayesowskich?

**Odpowiedź:**

**Twierdzenie Bayesa** pozwala zaktualizować przekonanie o hipotezie $\theta$ po zobaczeniu danych $D$:
$$P(\theta\mid D)=\frac{P(D\mid\theta)\,P(\theta)}{P(D)}$$
- $P(\theta)$ - **prior** (wiedza przed danymi),
- $P(D\mid\theta)$ - **likelihood** (wiarygodność),
- $P(\theta\mid D)$ - **posterior**,
- $P(D)=\sum_\theta P(D\mid\theta)P(\theta)$ - **evidence** (stała normalizująca).

#### Przykład liczbowy
Choroba występuje u 1% populacji, test ma czułość 95% i swoistość 90% (fałszywie dodatni 10%). Dla wyniku dodatniego:
$$P(\text{ch}\mid +)=\frac{0{,}95\cdot0{,}01}{0{,}95\cdot0{,}01+0{,}10\cdot0{,}99}=\frac{0{,}0095}{0{,}1085}\approx 8{,}8\%$$
Mały prior (base rate) dominuje - typowy błąd to ignorowanie częstości bazowej.

#### Naive Bayes
Klasyfikator wybiera klasę maksymalizującą posterior:
$$\hat y=\arg\max_y P(y)\prod_{j=1}^d P(x_j\mid y)$$
"Naiwne" założenie: cechy są **warunkowo niezależne** przy danej klasie. Dzięki temu estymujemy $d$ jednowymiarowych rozkładów zamiast jednego $d$-wymiarowego. Warianty: Gaussian, Multinomial (tekst, liczności słów), Bernoulli. Stosuje się wygładzanie Laplace'a, aby uniknąć zerowych prawdopodobieństw, oraz sumowanie logarytmów, by uniknąć underflow. Mimo nierealistycznego założenia daje dobre wyniki (np. spam), choć jego prawdopodobieństwa są słabo skalibrowane.

#### Szersze metody bayesowskie
- **MAP** i regularyzacja (prior Gaussa = L2, Laplace'a = L1),
- **regresja bayesowska**, procesy gaussowskie, **Bayesian optimization** (dobór hiperparametrów),
- **MCMC i variational inference** przy nieanalitycznym posteriorze,
- **Bayesian neural networks** i szacowanie niepewności,
- filtr Kalmana, aktualizacje online (posterior jako nowy prior).

**Źródła:**
- [Bayes' theorem (Wikipedia)](https://en.wikipedia.org/wiki/Bayes%27_theorem)
- [scikit-learn - Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Naive Bayes classifier (Wikipedia)](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

---

<a id="q312"></a>
### 312. Jaka jest różnica między estymacją MLE a MAP?

**Odpowiedź:**

Obie metody wyznaczają **punktowy** estymator parametrów $\theta$, ale różnią się użyciem wiedzy a priori.

#### MLE (Maximum Likelihood Estimation)
$$\hat\theta_{MLE}=\arg\max_\theta P(D\mid\theta)=\arg\max_\theta\sum_i\log p(x_i\mid\theta)$$
Nie zakłada niczego o $\theta$ poza modelem danych.

#### MAP (Maximum A Posteriori)
$$\hat\theta_{MAP}=\arg\max_\theta P(\theta\mid D)=\arg\max_\theta\Big[\sum_i\log p(x_i\mid\theta)+\log p(\theta)\Big]$$
(mianownik $P(D)$ nie zależy od $\theta$). To MLE z dodatkowym członem **prior**.

#### Związek z regularyzacją
- Prior Gaussa $\theta\sim N(0,\tau^2 I)$ daje $\log p(\theta)=-\frac{\|\theta\|^2}{2\tau^2}+c$, czyli **regularyzację L2** (ridge / weight decay).
- Prior Laplace'a daje **L1** (lasso, rzadkość).
- Płaski (jednostajny) prior sprowadza MAP do MLE.

#### Porównanie
| Cecha | MLE | MAP |
|---|---|---|
| Prior | brak | tak |
| Przy małej próbie | wysoka wariancja, ryzyko overfittingu | stabilniejszy dzięki priorowi |
| Przy dużej próbie | | zbiega do MLE (dane dominują prior) |
| Niezmienniczość na reparametryzację | tak | nie |
| Wynik | punkt | punkt (nie pełny posterior) |

#### Przykład: rzut monetą
Zaobserwowano 3 orły w 3 rzutach. MLE: $\hat p=1$ (absurdalne). MAP z priorem Beta$(2,2)$: $\hat p=\frac{3+2-1}{3+4-2}=\frac{4}{5}=0{,}8$ - rozsądniejsze. Pełne podejście bayesowskie zwraca cały rozkład $\text{Beta}(5,2)$.

Uwaga: MAP nadal jest estymatorem punktowym, więc nie daje niepewności; do tego potrzeba pełnego posterioru (MCMC, variational inference).

**Źródła:**
- [Maximum likelihood estimation (Wikipedia)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
- [Maximum a posteriori estimation (Wikipedia)](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation)
- [Deep Learning Book - rozdz. 5 Machine Learning Basics](https://www.deeplearningbook.org/contents/ml.html)

---
<a id="q313"></a>
### 313. Czym jest estymacja największej wiarygodności (Maximum Likelihood Estimation, MLE)?

**Odpowiedź:**

**MLE** wybiera parametry $\theta$, dla których zaobserwowane dane są **najbardziej prawdopodobne** w danym modelu. Dla niezależnych obserwacji $x_1,\dots,x_n$:
$$L(\theta)=\prod_{i=1}^n p(x_i\mid\theta),\qquad \ell(\theta)=\sum_{i=1}^n\log p(x_i\mid\theta),\qquad\hat\theta=\arg\max_\theta\ell(\theta)$$
Logarytm zamienia iloczyn na sumę (stabilność numeryczna, prostsze pochodne) i nie zmienia położenia maksimum. Rozwiązujemy $\nabla_\theta\ell=0$ analitycznie lub numerycznie (gradient descent, Newton, EM).

#### Przykłady
- **Rozkład normalny**: $\hat\mu=\bar x$, $\hat\sigma^2=\frac1n\sum(x_i-\bar x)^2$ (estymator obciążony; wersja z $n-1$ jest nieobciążona).
- **Bernoulli**: $\hat p=k/n$.
- **Regresja liniowa** z szumem gaussowskim: maksymalizacja likelihood = minimalizacja MSE (najmniejszych kwadratów).
- **Regresja logistyczna / klasyfikacja**: maksymalizacja likelihood = minimalizacja **cross-entropy** (negative log-likelihood).
- Trening sieci neuronowych i LLM (next-token prediction) to w istocie MLE.

#### Własności
- **Zgodność** (zbiega do prawdziwego $\theta$ przy $n\to\infty$), **asymptotyczna normalność** i efektywność (osiąga granicę Craméra-Rao) przy warunkach regularności.
- Niezmienniczość na reparametryzację.
- Wady: przy małym $n$ overfitting (np. $\hat p=1$ po 3 orłach w 3 rzutach), brak priorów (stąd MAP), możliwość braku jednoznacznego maksimum lub maksimów lokalnych, wrażliwość na błędnie dobrany model.

#### Fragment kodu
```python
import torch
x = torch.randn(1000)*2 + 5
mu = torch.zeros(1, requires_grad=True)
log_s = torch.zeros(1, requires_grad=True)
opt = torch.optim.Adam([mu, log_s], lr=0.05)
for _ in range(500):
    nll = -torch.distributions.Normal(mu, log_s.exp()).log_prob(x).sum()
    opt.zero_grad(); nll.backward(); opt.step()
print(mu.item(), log_s.exp().item())   # ~5, ~2
```

**Źródła:**
- [Maximum likelihood estimation (Wikipedia)](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
- [Deep Learning Book - rozdz. 5.5 Maximum Likelihood Estimation](https://www.deeplearningbook.org/contents/ml.html)
- [Likelihood function (Wikipedia)](https://en.wikipedia.org/wiki/Likelihood_function)

---

<a id="q314"></a>
### 314. Wyjaśnij podejście bayesowskie a częstościowe (Bayesian vs Frequentist) w statystyce.

**Odpowiedź:**

Różnica dotyczy **interpretacji prawdopodobieństwa** i tego, co uznajemy za losowe.

#### Podejście częstościowe (frequentist)
- Prawdopodobieństwo = granica częstości względnej w nieskończonej serii powtórzeń.
- Parametry $\theta$ są **stałe, lecz nieznane**; losowe są dane.
- Narzędzia: MLE, **p-value**, testy hipotez, **przedziały ufności** (pokrycie procedury).
- Nie przypisuje prawdopodobieństwa hipotezom ("prawdopodobieństwo, że $H_0$ jest prawdziwa" nie ma sensu).

#### Podejście bayesowskie
- Prawdopodobieństwo = **stopień przekonania**.
- Parametry są **zmiennymi losowymi** z rozkładem prior $p(\theta)$; po zobaczeniu danych: $p(\theta\mid D)\propto p(D\mid\theta)p(\theta)$.
- Narzędzia: posterior, **przedziały wiarygodności** (credible intervals), czynnik Bayesa, rozkład predykcyjny $p(y^*\mid D)=\int p(y^*\mid\theta)p(\theta\mid D)d\theta$.
- Naturalnie kwantyfikuje niepewność i pozwala łączyć wiedzę ekspercką z danymi.

#### Porównanie
| Aspekt | Częstościowe | Bayesowskie |
|---|---|---|
| Parametr | stały | rozkład |
| Wynik | estymata punktowa, CI, p-value | pełny posterior |
| Prior | brak | wymagany (subiektywność/wrażliwość) |
| Małe próby | słabsze | pomaga informatywny prior |
| Koszt obliczeń | zwykle niski | wysoki (MCMC, VI) |
| Interpretacja 95% przedziału | procedura pokrywa parametr w 95% powtórzeń | 95% prawdopodobieństwa, że $\theta$ leży w przedziale (przy modelu i priorze) |

#### W ML
- Częstościowe: klasyczny trening (MLE, regularyzacja jako heurystyka), testy A/B z p-value.
- Bayesowskie: procesy gaussowskie, Bayesian optimization, Bayesian NN, testy A/B z rozkładem posterior ("prawdopodobieństwo, że wariant B jest lepszy"), modele hierarchiczne.
- Oba podejścia zbiegają przy dużej ilości danych i słabym priorze; nie są wzajemnie wykluczające się - w praktyce używa się obu (MAP jako pomost).

**Źródła:**
- [Bayesian statistics (Wikipedia)](https://en.wikipedia.org/wiki/Bayesian_statistics)
- [Frequentist inference (Wikipedia)](https://en.wikipedia.org/wiki/Frequentist_inference)
- [Bayesian inference (Wikipedia)](https://en.wikipedia.org/wiki/Bayesian_inference)

---

<a id="q315"></a>
### 315. Czym jest centralne twierdzenie graniczne (Central Limit Theorem, CLT) i dlaczego jest ważne?

**Odpowiedź:**

**CLT**: jeśli $X_1,\dots,X_n$ są niezależne, o tym samym rozkładzie ze średnią $\mu$ i skończoną wariancją $\sigma^2$, to rozkład znormalizowanej średniej z próby zbiega do normalnego:
$$\frac{\bar X_n-\mu}{\sigma/\sqrt n}\xrightarrow{d}N(0,1)$$
Innymi słowy $\bar X_n\approx N(\mu,\sigma^2/n)$ **niezależnie od kształtu rozkładu źródłowego** (byle wariancja była skończona). Błąd standardowy średniej maleje jak $1/\sqrt n$.

#### Intuicja
Suma wielu małych, niezależnych wkładów daje kształt dzwonu - np. średnia z rzutów kostką dla $n=30$ jest już bliska normalnej, choć pojedynczy rzut ma rozkład jednostajny.

#### Dlaczego ważne
- Uzasadnia **testy t/z** i **przedziały ufności** dla średnich bez zakładania normalności danych.
- Podstawa wnioskowania w testach A/B (średnie i proporcje).
- Wyjaśnia, czemu szum jako suma wielu czynników bywa gaussowski (uzasadnienie założeń regresji, inicjalizacji wag).
- Pokazuje, jak szybko maleje niepewność metryk ewaluacji wraz z wielkością zbioru testowego.
- Podstawa bootstrapu i estymatorów Monte Carlo (błąd $\sim 1/\sqrt n$).

#### Ograniczenia
- Wymaga skończonej wariancji: dla rozkładu Cauchy'ego CLT **nie** działa.
- Zbieżność zależy od skośności i ciężkości ogonów (dla mocno skośnych danych potrzeba dużego $n$; reguła $n\ge30$ to tylko heurystyka).
- Zakłada niezależność; przy silnej autokorelacji (szeregi czasowe) efektywna liczba próbek jest mniejsza.

```python
import numpy as np
rng = np.random.default_rng(0)
means = rng.exponential(1.0, size=(10000, 50)).mean(axis=1)
print(means.mean(), means.std())   # ~1.0 oraz ~1/sqrt(50)=0.141
```

**Źródła:**
- [Central limit theorem (Wikipedia)](https://en.wikipedia.org/wiki/Central_limit_theorem)
- [Law of large numbers (Wikipedia)](https://en.wikipedia.org/wiki/Law_of_large_numbers)
- [Deep Learning Book - rozdz. 3 Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html)

---

<a id="q316"></a>
### 316. Czym są techniki próbkowania (sampling techniques)?

**Odpowiedź:**

**Próbkowanie** to wybór podzbioru danych z populacji tak, aby wnioski z próby dało się uogólnić. W ML służy do budowy zbiorów treningowych/testowych, redukcji kosztów, radzenia sobie z niezbalansowaniem oraz w metodach probabilistycznych.

#### Próbkowanie probabilistyczne
- **Proste losowe (SRS)** - każdy element ma takie samo prawdopodobieństwo; ze zwracaniem lub bez.
- **Warstwowe (stratified)** - populacja dzielona na warstwy (np. klasy), próbkowanie proporcjonalne w każdej; zachowuje rozkład klas, kluczowe przy niezbalansowanych danych (`train_test_split(..., stratify=y)`, `StratifiedKFold`).
- **Systematyczne** - co $k$-ty element z listy (uwaga na ukryte okresowości).
- **Klastrowe (cluster)** - losujemy całe grupy (np. szkoły, urządzenia), taniej, lecz z większą wariancją.
- **Reservoir sampling** - jednoprzebiegowe losowanie $k$ elementów ze strumienia o nieznanej długości.

#### Nieprobabilistyczne
- Wygodne, kwotowe, celowe, kuli śnieżnej - szybkie, ale obciążone (selection bias); do wnioskowania statystycznego nie nadają się.

#### Próbkowanie w ML
- **Under/oversampling** klas (random oversampling, **SMOTE**), ważenie klas jako alternatywa.
- **Importance sampling** - próbkowanie z rozkładu pomocniczego i ważenie $w=p/q$ (RL off-policy, Monte Carlo).
- **Negative sampling** (word2vec, systemy rekomendacyjne), **hard negative mining**.
- **MCMC / Gibbs / Metropolis-Hastings** - próbkowanie z posteriora.
- **Próbkowanie z modelu** (temperature, top-k, top-p w LLM).
- **Mini-batch sampling** w SGD.

#### Pułapki
- **Selection/survivorship bias** - próba nie reprezentuje populacji.
- **Wyciek danych** - losowy podział szeregów czasowych lub wielu wierszy tego samego użytkownika; stosuj podział czasowy lub grupowy (`GroupKFold`).
- Oversampling przed splitem powoduje leakage - próbkuj tylko zbiór treningowy.
- Błąd próbkowania maleje jak $1/\sqrt n$, nie jak $1/n$.

**Źródła:**
- [Sampling (statistics) (Wikipedia)](https://en.wikipedia.org/wiki/Sampling_(statistics))
- [scikit-learn - Cross-validation iterators (Stratified, Group)](https://scikit-learn.org/stable/modules/cross_validation.html)
- [imbalanced-learn - Over-sampling](https://imbalanced-learn.org/stable/over_sampling.html)
- [Reservoir sampling (Wikipedia)](https://en.wikipedia.org/wiki/Reservoir_sampling)

---

<a id="q317"></a>
### 317. Czym jest metoda bootstrap i jak się ją stosuje?

**Odpowiedź:**

**Bootstrap** (Efron, 1979) to metoda estymacji rozkładu statystyki przez **wielokrotne próbkowanie ze zwracaniem** z zaobserwowanej próby. Zakładamy, że empiryczny rozkład próby przybliża populację, więc losowanie z niej symuluje losowanie z populacji.

#### Algorytm
1. Z próby o rozmiarze $n$ losuj ze zwracaniem $n$ elementów (próba bootstrapowa).
2. Oblicz statystykę $\hat\theta^*_b$ (średnia, mediana, AUC, współczynnik modelu).
3. Powtórz $B$ razy (zwykle 1000-10000).
4. Rozkład $\{\hat\theta^*_b\}$ przybliża rozkład próbkowy: oszacuj błąd standardowy, obciążenie, przedział ufności (np. **percentylowy** 2,5-97,5 centyl).

Każda próba bootstrapowa zawiera średnio ok. $1-(1-1/n)^n\to63{,}2\%$ unikalnych obserwacji; pozostałe ~36,8% to **out-of-bag (OOB)**.

#### Zastosowania
- **Przedziały ufności** dla dowolnych metryk (F1, AUC), gdy nie ma wzoru analitycznego.
- **Porównanie modeli** - bootstrap różnicy metryk na zbiorze testowym.
- **Bagging i Random Forest** - trening wielu modeli na próbach bootstrapowych + agregacja; ocena **OOB** bez osobnego zbioru walidacyjnego.
- Ocena niepewności współczynników i ważności cech.

```python
import numpy as np
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
def boot_auc(y, s, B=2000):
    n, out = len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():          # potrzebne obie klasy
            out.append(roc_auc_score(y[i], s[i]))
    return np.percentile(out, [2.5, 97.5])
```

#### Ograniczenia
- Wymaga danych i.i.d. (dla szeregów czasowych - block bootstrap).
- Zawodzi dla statystyk skrajnych (max, min) i przy bardzo małych próbach.
- Kosztowne obliczeniowo przy dużych modelach (retrening $B$ razy).
- Nie naprawia obciążonej próby - odwzorowuje jej wady.

**Źródła:**
- [Bootstrapping (Wikipedia)](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [Breiman (1996): Bagging predictors (Machine Learning)](https://link.springer.com/article/10.1007/BF00058655)
- [scikit-learn - Ensemble methods (bagging, OOB)](https://scikit-learn.org/stable/modules/ensemble.html)

---

## Kodowanie

<a id="q318"></a>
### 318. Napisz funkcję w Pythonie obliczającą błąd średniokwadratowy (MSE).

**Odpowiedź:**

$$\text{MSE}=\frac1n\sum_{i=1}^n(y_i-\hat y_i)^2$$
MSE jest średnią kwadratów reszt: silnie karze duże błędy (outliery), jest różniczkowalna, a jej gradient po predykcji to $\frac{2}{n}(\hat y-y)$. Jednostką jest kwadrat jednostki $y$, dlatego często raportuje się RMSE $=\sqrt{\text{MSE}}$.

#### Implementacja (NumPy)
```python
import numpy as np

def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Kształty y_true i y_pred muszą być zgodne")
    return np.mean((y_true - y_pred) ** 2)

print(mse([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]))   # 0.375
```
Sprawdzenie ręczne: reszty $0{,}5,-0{,}5,0,-1$, kwadraty $0{,}25+0{,}25+0+1=1{,}5$, po podzieleniu przez 4 daje $0{,}375$.

#### Czysty Python
```python
def mse_py(y_true, y_pred):
    assert len(y_true) == len(y_pred) and len(y_true) > 0
    return sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / len(y_true)
```

#### Wersja z gradientem
```python
def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
```

#### Uwagi
- Weryfikuj z `sklearn.metrics.mean_squared_error` (parametr `squared=False` lub `root_mean_squared_error` daje RMSE).
- Dla wielu wyjść stosuj `axis=0` i uśrednij lub zwróć wektor.
- MSE odpowiada MLE przy szumie gaussowskim; wrażliwa na outliery - alternatywy: MAE, Huber loss.
- Skala zależy od danych, więc MSE nie porównuje się między zbiorami; użyj $R^2$ lub znormalizowanych metryk.
- Zadbaj o typ float (dla `uint8` odejmowanie zawija się!).

**Źródła:**
- [Implement Mean Squared Error (MSE) cost function (build-your-own-x-machine-learning)](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning/blob/main/tutorials/core-machine-learning-algorithms/mean-squared-error/mean_squared_error.py)
- [scikit-learn - mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)
- [Mean squared error (Wikipedia)](https://en.wikipedia.org/wiki/Mean_squared_error)

---

<a id="q319"></a>
### 319. Napisz funkcję w Pythonie obliczającą średni błąd bezwzględny (MAE).

**Odpowiedź:**

$$\text{MAE}=\frac1n\sum_{i=1}^n|y_i-\hat y_i|$$
MAE ma tę samą jednostkę co $y$, jest łatwa do interpretacji ("średnio mylimy się o X") i **odporna na outliery** w porównaniu z MSE (kara liniowa, nie kwadratowa). Jest minimalizowana przez **medianę** warunkową (MSE - przez średnią). Nie jest różniczkowalna w 0 (używa się subgradientu, $\text{sign}(\hat y-y)/n$) i ma stały gradient, co bywa wolniejsze w optymalizacji przy końcu treningu - kompromis to Huber loss.

#### Implementacja
```python
import numpy as np

def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("Kształty muszą być zgodne")
    return np.mean(np.abs(y_true - y_pred))

print(mae([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]))   # 0.5
```
Ręcznie: $|0{,}5|+|0{,}5|+0+|1|=2$, po podzieleniu przez 4 daje $0{,}5$.

#### Subgradient
```python
def mae_grad(y_true, y_pred):
    return np.sign(y_pred - y_true) / y_true.size
```

#### Porównanie z MSE
| | MAE | MSE/RMSE |
|---|---|---|
| Wrażliwość na outliery | niska | wysoka |
| Optymalny predyktor | mediana | średnia |
| Jednostka | jak $y$ | kwadrat (RMSE: jak $y$) |
| Gradient | stały, nieciągły | proporcjonalny do błędu |

#### Uwagi
- Zawsze $\text{MAE}\le\text{RMSE}$; duża różnica wskazuje na obecność dużych błędów.
- Weryfikacja: `sklearn.metrics.mean_absolute_error`.
- Wersje względne: MAPE (problem przy $y=0$), sMAPE, MASE (szeregi czasowe).
- Wybór metryki powinien odpowiadać kosztowi biznesowemu błędów.

**Źródła:**
- [Implement Mean Absolute Error (MAE) cost function (build-your-own-x-machine-learning)](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning/blob/main/tutorials/core-machine-learning-algorithms/mean-absolute-error/mean_absolute_error.py)
- [scikit-learn - mean_absolute_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)
- [Mean absolute error (Wikipedia)](https://en.wikipedia.org/wiki/Mean_absolute_error)

---
<a id="q320"></a>
### 320. Zaimplementuj prosty model regresji liniowej od zera.

**Odpowiedź:**

Model: $\hat y=Xw+b$. Funkcja kosztu: $J=\frac1n\|Xw+b-y\|^2$. Gradienty:
$$\frac{\partial J}{\partial w}=\frac2n X^\top(\hat y-y),\qquad\frac{\partial J}{\partial b}=\frac2n\sum(\hat y-y)$$
Dwie drogi: **gradient descent** (skaluje się na duże dane) lub **równanie normalne** $w=(X^\top X)^{-1}X^\top y$ (rozwiązanie zamknięte, koszt $O(d^3)$, wrażliwe na współliniowość).

#### Implementacja
```python
import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr, self.n_iters = lr, n_iters
        self.w, self.b = None, 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.n_iters):
            err = X @ self.w + self.b - y
            self.w -= self.lr * (2 / n) * (X.T @ err)
            self.b -= self.lr * (2 / n) * err.sum()
        return self

    def predict(self, X):
        return X @ self.w + self.b

    # wariant zamknięty (z kolumną jedynek na bias)
    def fit_normal(self, X, y):
        Xb = np.c_[np.ones(len(X)), X]
        theta = np.linalg.lstsq(Xb, y, rcond=None)[0]   # stabilniej niż inv
        self.b, self.w = theta[0], theta[1:]
        return self

# test
rng = np.random.default_rng(0)
X = rng.normal(size=(200, 3))
y = X @ np.array([2., -1., .5]) + 3 + rng.normal(scale=.1, size=200)
m = LinearRegression(lr=0.1, n_iters=500).fit(X, y)
print(m.w, m.b)    # ~[2, -1, .5], ~3
```

#### Wskazówki
- **Skaluj cechy** (standaryzacja) - inaczej gradient descent zbiega wolno lub się rozbiega; dobierz learning rate.
- Do rozwiązania zamkniętego używaj `lstsq`/`solve`, nie jawnej odwrotności.
- Regularyzacja L2 (ridge): dodaj $\lambda w$ do gradientu (bias nie jest regularyzowany).
- Monitoruj loss w trakcie treningu; sprawdź wynik względem `sklearn.linear_model.LinearRegression`.
- Założenia OLS: liniowość, niezależność błędów, homoskedastyczność, brak silnej współliniowości.

**Źródła:**
- [Implement Linear Regression from scratch (build-your-own-x-machine-learning)](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning/blob/main/tutorials/core-machine-learning-algorithms/linear-regression/linear_regression.py)
- [scikit-learn - Linear Models (OLS)](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
- [Linear regression (Wikipedia)](https://en.wikipedia.org/wiki/Linear_regression)

---

<a id="q321"></a>
### 321. Zaimplementuj prosty model regresji logistycznej od zera.

**Odpowiedź:**

Model: $p=\sigma(z)$, $z=Xw+b$, $\sigma(z)=\frac1{1+e^{-z}}$. Loss to **binary cross-entropy** (ujemny log-likelihood):
$$J=-\frac1n\sum\big[y\log p+(1-y)\log(1-p)\big]$$
Elegancki gradient (ten sam kształt co w regresji liniowej):
$$\frac{\partial J}{\partial w}=\frac1n X^\top(p-y),\qquad\frac{\partial J}{\partial b}=\frac1n\sum(p-y)$$
Funkcja kosztu jest wypukła, więc gradient descent znajduje optimum globalne.

#### Implementacja
```python
import numpy as np

def sigmoid(z):
    # stabilna numerycznie
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

class LogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000, l2=0.0):
        self.lr, self.n_iters, self.l2 = lr, n_iters, l2

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.n_iters):
            p = sigmoid(X @ self.w + self.b)
            self.w -= self.lr * (X.T @ (p - y) / n + self.l2 * self.w)
            self.b -= self.lr * (p - y).mean()
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, thr=0.5):
        return (self.predict_proba(X) >= thr).astype(int)

def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
```

#### Wskazówki
- Standaryzuj cechy; przy idealnej separowalności wagi rosną w nieskończoność - pomaga regularyzacja L2.
- Clipping prawdopodobieństw chroni przed $\log 0$.
- Próg 0,5 nie jest święty - dobierz go do kosztu FP/FN (krzywa PR/ROC); przy niezbalansowanych klasach użyj wag klas.
- Wielu klas: softmax + cross-entropy.
- Weryfikacja: porównaj z `sklearn.linear_model.LogisticRegression` (uwaga: domyślnie ma regularyzację, $C=1$).
- Wagi interpretuje się przez **log-odds**: $e^{w_j}$ to mnożnik szans przy wzroście cechy o 1.

**Źródła:**
- [Implement Logistic Regression from scratch (build-your-own-x-machine-learning)](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning/blob/main/tutorials/core-machine-learning-algorithms/logistic-regression/logistic_regression.py)
- [scikit-learn - Logistic regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- [Logistic regression (Wikipedia)](https://en.wikipedia.org/wiki/Logistic_regression)

---

<a id="q322"></a>
### 322. Zaimplementuj algorytm K najbliższych sąsiadów (K-Nearest Neighbors, KNN).

**Odpowiedź:**

KNN to metoda **leniwa** (lazy, instance-based): brak fazy treningu, tylko zapamiętanie danych. Predykcja dla punktu $x$: znajdź $k$ najbliższych przykładów treningowych (według metryki, zwykle euklidesowej $d(a,b)=\|a-b\|_2$) i zagłosuj większością (klasyfikacja) lub uśrednij (regresja).

#### Implementacja (klasyfikacja, wektoryzowana)
```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        # macierz odległości (m x n) bez pętli: ||a-b||^2 = a^2 + b^2 - 2ab
        d2 = (X**2).sum(1)[:, None] + (self.X**2).sum(1)[None, :] - 2 * X @ self.X.T
        idx = np.argpartition(d2, self.k, axis=1)[:, :self.k]   # O(n) na zapytanie
        return np.array([Counter(self.y[i]).most_common(1)[0][0] for i in idx])

rng = np.random.default_rng(0)
Xa = rng.normal(0, 1, (50, 2)); Xb = rng.normal(3, 1, (50, 2))
X = np.vstack([Xa, Xb]); y = np.array([0]*50 + [1]*50)
print(KNN(3).fit(X, y).predict([[0, 0], [3, 3]]))   # [0 1]
```
(`argpartition` wymaga $k<n$.) Wersja regresyjna: `self.y[i].mean(axis=1)`.

#### Złożoność
- Trening $O(1)$, pamięć $O(nd)$; predykcja brute-force $O(nd)$ na zapytanie. Przyspieszenie: **KD-tree / ball tree** (dobre dla małego $d$), przybliżone ANN (FAISS, HNSW) dla dużej skali.

#### Wskazówki
- **Skalowanie cech jest obowiązkowe** (odległości zdominowane przez cechy o dużej skali).
- Wybór $k$ przez cross-validation; małe $k$ = wysoka wariancja, duże $k$ = wysokie obciążenie; nieparzyste $k$ przy 2 klasach (unikanie remisów).
- Ważenie głosów odwrotnością odległości.
- **Przekleństwo wymiarowości** - w wysokich wymiarach odległości tracą sens; użyj redukcji wymiarowości.
- Wrażliwy na niezbalansowanie klas.

**Źródła:**
- [Implement K-Nearest Neighbors (KNN) (build-your-own-x-machine-learning)](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning/blob/main/tutorials/core-machine-learning-algorithms/knn/knn.py)
- [scikit-learn - Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [k-nearest neighbors algorithm (Wikipedia)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

---

<a id="q323"></a>
### 323. Zaimplementuj funkcje aktywacji Sigmoid, Tanh, ReLU, LeakyReLU i Softmax.

**Odpowiedź:**

| Funkcja | Wzór | Zakres | Pochodna |
|---|---|---|---|
| Sigmoid | $\frac1{1+e^{-x}}$ | $(0,1)$ | $\sigma(1-\sigma)$ |
| Tanh | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $(-1,1)$ | $1-\tanh^2$ |
| ReLU | $\max(0,x)$ | $[0,\infty)$ | $1$ dla $x>0$, $0$ w p.p. |
| LeakyReLU | $x$ dla $x>0$, $\alpha x$ w p.p. | $\mathbb R$ | $1$ lub $\alpha$ |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | simplex (suma 1) | Jacobian $s_i(\delta_{ij}-s_j)$ |

#### Implementacja (NumPy)
```python
import numpy as np

def sigmoid(x):
    # stabilnie: unikamy exp(duże dodatnie)
    return np.where(x >= 0, 1/(1+np.exp(-np.abs(x))), np.exp(-np.abs(x))/(1+np.exp(-np.abs(x))))

def tanh(x):
    return np.tanh(x)            # lub 2*sigmoid(2x) - 1

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x, axis=-1):
    z = x - np.max(x, axis=axis, keepdims=True)   # stabilność numeryczna
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)

print(softmax(np.array([1000., 1001., 1002.])))   # bez odjęcia maksimum byłoby NaN
```
Softmax jest niezmienniczy na dodanie stałej do wszystkich wejść, więc odjęcie maksimum jest bezpieczne.

#### Kiedy co
- **Sigmoid**: wyjście binarne (prawdopodobieństwo), bramki w LSTM; w warstwach ukrytych wypiera go ReLU z powodu **zanikającego gradientu** i braku centrowania na zero.
- **Tanh**: wyjście wyśrodkowane na 0, nadal saturuje.
- **ReLU**: domyślny wybór (tani, brak saturacji dla $x>0$); ryzyko "martwych neuronów".
- **LeakyReLU**: niewielki gradient dla $x<0$ łagodzi ten problem.
- **Softmax**: wyjście wieloklasowe; w praktyce łączony z cross-entropy (np. `nn.CrossEntropyLoss` przyjmuje logity, więc softmax nie dodajemy ręcznie).

**Źródła:**
- [Implement Sigmoid, Tanh, ReLU, LeakyReLU, and Softmax Activation Functions (build-your-own-x-machine-learning)](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning/blob/main/tutorials/core-machine-learning-algorithms/activation-functions/activation_functions.py)
- [Activation function (Wikipedia)](https://en.wikipedia.org/wiki/Activation_function)
- [PyTorch - Non-linear activations](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)
- [Softmax function (Wikipedia)](https://en.wikipedia.org/wiki/Softmax_function)

---

<a id="q324"></a>
### 324. Jak zaimplementowałbyś klasteryzację k-means?

**Odpowiedź:**

**k-means** dzieli dane na $k$ klastrów, minimalizując sumę kwadratów odległości do centroidów (inertia):
$$J=\sum_{i=1}^n\|x_i-\mu_{c(i)}\|^2$$

#### Algorytm Lloyda
1. Zainicjalizuj $k$ centroidów (najlepiej **k-means++**).
2. **Przypisanie**: każdy punkt do najbliższego centroidu.
3. **Aktualizacja**: centroid = średnia punktów klastra.
4. Powtarzaj do zbieżności (brak zmian przypisań lub $\|\Delta\mu\|<\text{tol}$) lub limitu iteracji.
Każdy krok nie zwiększa $J$, więc algorytm zbiega, lecz do **minimum lokalnego**.

#### Implementacja
```python
import numpy as np

def kmeans(X, k, n_iter=100, tol=1e-6, seed=0):
    rng = np.random.default_rng(seed)
    # k-means++
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min([((X - c) ** 2).sum(1) for c in C], axis=0)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)

    for _ in range(n_iter):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)   # (n, k)
        labels = d.argmin(1)
        newC = np.array([X[labels == j].mean(0) if np.any(labels == j)
                         else X[rng.integers(len(X))] for j in range(k)])
        if np.linalg.norm(newC - C) < tol:
            C = newC; break
        C = newC
    inertia = ((X - C[labels]) ** 2).sum()
    return labels, C, inertia
```
Pusty klaster obsługujemy ponownym wylosowaniem centroidu.

#### Praktyka
- Uruchom kilka razy z różnymi inicjalizacjami (`n_init`) i wybierz najniższą inertię.
- **Wybór $k$**: metoda łokcia, silhouette score, gap statistic.
- **Skaluj cechy**; k-means zakłada kuliste klastry o podobnym rozmiarze i wariancji, jest wrażliwy na outliery (alternatywy: k-medoids, GMM, DBSCAN).
- Złożoność $O(nkdT)$; dla dużych danych `MiniBatchKMeans`.
- Sklearn: `KMeans(n_clusters=k, n_init=10).fit(X)`.

**Źródła:**
- [scikit-learn - Clustering: K-means](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Arthur & Vassilvitskii (2007): k-means++ (Stanford)](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf)
- [k-means clustering (Wikipedia)](https://en.wikipedia.org/wiki/K-means_clustering)

---

<a id="q325"></a>
### 325. Napisz kod wykonujący k-krotną walidację krzyżową (k-fold cross-validation).

**Odpowiedź:**

**K-fold CV**: dzielimy dane na $k$ rozłącznych foldów; $k$ razy trenujemy na $k-1$ foldach i oceniamy na pozostałym; wynik to średnia (i odchylenie) metryk. Daje bardziej stabilną ocenę niż pojedynczy podział i wykorzystuje wszystkie dane zarówno do treningu, jak i walidacji. Typowo $k=5$ lub $10$.

#### Implementacja od zera
```python
import numpy as np

def kfold_indices(n, k=5, shuffle=True, seed=0):
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val

def cross_val_score(make_model, X, y, metric, k=5):
    scores = []
    for tr, va in kfold_indices(len(X), k):
        model = make_model()                 # świeży model w każdym foldzie
        model.fit(X[tr], y[tr])
        scores.append(metric(y[va], model.predict(X[va])))
    return np.mean(scores), np.std(scores)
```

#### Wersja scikit-learn (zalecana, bez leakage)
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = make_pipeline(StandardScaler(), LogisticRegression())   # scaler dopasowany tylko na train foldu
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
print(scores.mean(), scores.std())
```

#### Pułapki i warianty
- **Wyciek danych**: preprocessing (skalowanie, imputacja, selekcja cech) musi być fitowany wewnątrz foldu - użyj `Pipeline`.
- **Stratified** dla klasyfikacji (zachowuje proporcje klas); **GroupKFold** gdy wiele wierszy dotyczy tej samej jednostki; **TimeSeriesSplit** dla szeregów czasowych (bez tasowania).
- **Nested CV** przy jednoczesnym doborze hiperparametrów i ocenie.
- Większe $k$ = mniejsze obciążenie, ale wyższy koszt (LOOCV skrajnie); zbiór testowy końcowy trzymaj osobno.

**Źródła:**
- [scikit-learn - Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html)
- [scikit-learn - KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)
- [Cross-validation (Wikipedia)](https://en.wikipedia.org/wiki/Cross-validation_(statistics))

---

<a id="q326"></a>
### 326. Jak użyjesz Pandas do wczytania i oczyszczenia danych?

**Odpowiedź:**

Typowy przepływ: **wczytaj -> zbadaj -> oczyść -> zwaliduj -> zapisz**, najlepiej jako powtarzalny skrypt/pipeline.

#### 1. Wczytanie
```python
import pandas as pd, numpy as np

df = pd.read_csv("data.csv", parse_dates=["date"], na_values=["", "NA", "?", "-"],
                 dtype={"user_id": "string"})    # też read_parquet, read_json, read_sql
```
Przy dużych plikach: `usecols`, `chunksize`, dtypes typu `category`, format Parquet.

#### 2. Eksploracja
```python
df.info(); df.describe(include="all")
df.isna().mean().sort_values(ascending=False)   # odsetek braków
df.duplicated().sum(); df.nunique()
```

#### 3. Czyszczenie
```python
df = df.drop_duplicates()
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# typy
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["country"] = df["country"].str.strip().str.upper().astype("category")

# braki: numeryczne - mediana, kategoryczne - "unknown" lub moda
df["age"] = df["age"].fillna(df["age"].median())
df["city"] = df["city"].fillna("unknown")
df = df.dropna(subset=["target"])                # brak etykiety = usuń

# outliery (IQR)
q1, q3 = df["price"].quantile([.25, .75]); iqr = q3 - q1
df = df[df["price"].between(q1 - 1.5*iqr, q3 + 1.5*iqr)]

# reguły spójności
df = df[(df["age"] >= 0) & (df["age"] <= 120)]
```

#### 4. Walidacja i zapis
```python
assert df["id"].is_unique
df.to_parquet("clean.parquet", index=False)
```

#### Dobre praktyki
- Nie modyfikuj surowych danych; zachowaj kod jako funkcję/pipeline (`df.pipe(...)`, łańcuchy metod).
- Statystyki do imputacji (mediana, średnia) licz **tylko na zbiorze treningowym**, aby uniknąć leakage - w ML lepiej użyć `SimpleImputer` w `Pipeline`.
- Loguj, ile wierszy usunięto i dlaczego; uważaj na `SettingWithCopyWarning` (używaj `.loc`).
- Dla dużych danych rozważ Polars/Dask.

**Źródła:**
- [pandas - IO tools (read_csv, read_parquet)](https://pandas.pydata.org/docs/user_guide/io.html)
- [pandas - Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [scikit-learn - Imputation of missing values](https://scikit-learn.org/stable/modules/impute.html)

---
<a id="q327"></a>
### 327. Zaimplementuj k-najbliższych sąsiadów (KNN) od zera.

**Odpowiedź:**

To wariant pytania o implementację KNN - poniżej wersja **nieco inna niż wektoryzowana**: czytelna, z obsługą wielu metryk, ważenia odległością i regresji, w stylu oczekiwanym na rozmowie ("od zera", bez sklearn).

#### Algorytm
1. Zapamiętaj $(X,y)$.
2. Dla zapytania $x$ policz odległość do każdego punktu treningowego.
3. Wybierz $k$ najmniejszych.
4. Klasyfikacja: głosowanie (opcjonalnie ważone $1/d$); regresja: (ważona) średnia.

#### Implementacja
```python
import numpy as np
from collections import defaultdict

class KNNClassifier:
    def __init__(self, k=3, metric="euclidean", weighted=False):
        self.k, self.metric, self.weighted = k, metric, weighted

    def _dist(self, a, B):
        if self.metric == "euclidean":
            return np.sqrt(((B - a) ** 2).sum(axis=1))
        if self.metric == "manhattan":
            return np.abs(B - a).sum(axis=1)
        if self.metric == "cosine":
            return 1 - (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-12)
        raise ValueError(self.metric)

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y)
        return self

    def _predict_one(self, x):
        d = self._dist(x, self.X)
        nn = np.argsort(d)[: self.k]
        votes = defaultdict(float)
        for i in nn:
            votes[self.y[i]] += 1 / (d[i] + 1e-9) if self.weighted else 1
        return max(votes, key=votes.get)

    def predict(self, X):
        return np.array([self._predict_one(x) for x in np.asarray(X, float)])

    def score(self, X, y):
        return (self.predict(X) == np.asarray(y)).mean()
```

#### Na co zwrócić uwagę
- **Złożoność** predykcji $O(nd + n\log n)$ na zapytanie (`argsort`); `np.argpartition` obniża do $O(n)$; drzewa KD/ball tree lub ANN dla dużych zbiorów.
- **Skalowanie cech** oraz wybór metryki (cosine dla tekstu/embeddingów).
- Dobór $k$ przez CV; rozstrzyganie remisów (mniejsze $k$, wagi odległości).
- Wyciek: przy ocenie na zbiorze treningowym najbliższym sąsiadem jest sam punkt (dla $k=1$ accuracy = 100%) - oceniaj na osobnych danych lub leave-one-out.
- Przekleństwo wymiarowości, wrażliwość na cechy nieistotne.
- Regresja: `np.average(self.y[nn], weights=1/(d[nn]+1e-9))`.

**Źródła:**
- [scikit-learn - Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [Implement K-Nearest Neighbors (KNN) (build-your-own-x-machine-learning)](https://github.com/amitshekhariitbhu/build-your-own-x-machine-learning/blob/main/tutorials/core-machine-learning-algorithms/knn/knn.py)
- [k-nearest neighbors algorithm (Wikipedia)](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)

---

<a id="q328"></a>
### 328. Napisz kod obliczający precision i recall.

**Odpowiedź:**

Z macierzy pomyłek (TP, FP, FN, TN):
$$\text{precision}=\frac{TP}{TP+FP},\qquad\text{recall}=\frac{TP}{TP+FN},\qquad F_1=\frac{2PR}{P+R}$$
- **Precision**: spośród przewidzianych pozytywów, ile jest prawdziwych (koszt fałszywych alarmów - np. filtr spamu).
- **Recall** (czułość): spośród prawdziwych pozytywów, ile wykryto (koszt przeoczeń - np. diagnostyka medyczna, fraud).

#### Implementacja (binarna)
```python
import numpy as np

def precision_recall(y_true, y_pred, pos_label=1):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == pos_label) & (y_true == pos_label))
    fp = np.sum((y_pred == pos_label) & (y_true != pos_label))
    fn = np.sum((y_pred != pos_label) & (y_true == pos_label))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # zero_division
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

y_true = [1, 0, 1, 1, 0, 1, 0, 0]
y_pred = [1, 0, 0, 1, 1, 1, 0, 0]
print(precision_recall(y_true, y_pred))   # (0.75, 0.75)
```
Ręcznie: TP=3, FP=1, FN=1, więc $P=3/4$, $R=3/4$.

#### Wieloklasowe
```python
def macro_precision_recall(y_true, y_pred):
    classes = np.unique(y_true)
    ps, rs = zip(*[precision_recall(y_true, y_pred, c) for c in classes])
    return np.mean(ps), np.mean(rs)     # macro; micro = globalne TP/FP/FN
```
- **macro**: średnia po klasach (traktuje klasy równo), **micro**: sumowanie TP/FP/FN globalnie (dominują duże klasy), **weighted**: średnia ważona liczebnością.

#### Weryfikacja i uwagi
```python
from sklearn.metrics import precision_score, recall_score
precision_score(y_true, y_pred), recall_score(y_true, y_pred)
```
- Obsłuż dzielenie przez zero (brak przewidzianych pozytywów).
- Precision i recall zależą od **progu decyzyjnego** - przesunięcie progu zamienia jedno na drugie (krzywa PR, AUPRC).
- Przy niezbalansowanych klasach accuracy wprowadza w błąd; precision/recall/F1 są bardziej informatywne.

**Źródła:**
- [scikit-learn - Precision, recall and F-measures](https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics)
- [scikit-learn - precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)
- [Precision and recall (Wikipedia)](https://en.wikipedia.org/wiki/Precision_and_recall)

---

## Matematyka

<a id="q329"></a>
### 329. Wartości własne i wektory własne (Eigenvalues and Eigenvectors)

**Odpowiedź:**

Dla macierzy kwadratowej $A\in\mathbb R^{n\times n}$ niezerowy wektor $v$ jest **wektorem własnym**, a skalar $\lambda$ **wartością własną**, jeśli:
$$Av=\lambda v$$
Transformacja $A$ tylko **skaluje** $v$ (o $\lambda$; ujemne odwraca kierunek), nie zmieniając jego prostej. Wartości własne to pierwiastki **równania charakterystycznego** $\det(A-\lambda I)=0$; wektory własne to jądro $(A-\lambda I)$.

#### Przykład
$A=\begin{pmatrix}2&0\\0&3\end{pmatrix}$: $\lambda_1=2$, $v_1=(1,0)$; $\lambda_2=3$, $v_2=(0,1)$. Dla $B=\begin{pmatrix}2&1\\1&2\end{pmatrix}$: $\det(B-\lambda I)=(2-\lambda)^2-1=0\Rightarrow\lambda=1,3$, wektory $(1,-1)$ i $(1,1)$.

#### Własności
- $\text{tr}(A)=\sum\lambda_i$, $\det(A)=\prod\lambda_i$.
- **Macierz symetryczna** (np. kowariancji): wartości rzeczywiste, wektory własne ortogonalne; rozkład spektralny $A=Q\Lambda Q^\top$.
- Macierz symetryczna jest dodatnio półokreślona wtedy i tylko wtedy, gdy wszystkie $\lambda_i\ge0$.
- Diagonalizacja: $A=V\Lambda V^{-1}$; potęgi $A^k=V\Lambda^kV^{-1}$.

#### Zastosowania w ML
- **PCA**: wektory własne macierzy kowariancji to kierunki głównych składowych, a wartości własne to wariancje wzdłuż nich; wyjaśniona wariancja $=\lambda_i/\sum\lambda_j$.
- **SVD** (blisko spokrewnione): wartości osobliwe = pierwiastki wartości własnych $A^\top A$.
- **Spectral clustering** (wektory własne laplasjanu grafu), **PageRank** (dominujący wektor własny), analiza **Hessianu** (krzywizna, punkty siodłowe), stabilność dynamiki (promień spektralny, eksplodujące/zanikające gradienty w RNN), uwarunkowanie (stosunek $\lambda_{max}/\lambda_{min}$).

```python
import numpy as np
C = np.cov(X, rowvar=False)
vals, vecs = np.linalg.eigh(C)         # eigh dla macierzy symetrycznych
order = vals.argsort()[::-1]
vals, vecs = vals[order], vecs[:, order]
X_pca = (X - X.mean(0)) @ vecs[:, :2]   # rzut na 2 główne składowe
```
Używaj `eigh` (stabilna, wartości rzeczywiste) dla macierzy symetrycznych i `eig` dla ogólnych (mogą być zespolone).

**Źródła:**
- [Understanding Eigenvalues and Eigenvectors (Amit Shekhar)](https://x.com/amitiitbhu/status/1955895389225160877)
- [Eigenvalues and eigenvectors (Wikipedia)](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)
- [Deep Learning Book - rozdz. 2 Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)
- [NumPy - numpy.linalg.eigh](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html)

---

## Pytania behawioralne i scenariuszowe

<a id="q330"></a>
### 330. Opisz sytuację, w której poprawiłeś/aś wydajność modelu.

**Odpowiedź:**

To pytanie behawioralne - odpowiadaj w strukturze **STAR** (Situation, Task, Action, Result), z konkretnymi liczbami i pokazaniem metodycznego podejścia, nie "magicznej sztuczki".

#### Przykładowa odpowiedź
**Situation:** W poprzednim zespole odpowiadałem za model przewidujący churn klientów subskrypcyjnych. Wersja produkcyjna (gradient boosting) miała AUC-PR ok. 0,31 i zespół retencji narzekał, że lista klientów do kontaktu zawiera zbyt wielu fałszywych alarmów.

**Task:** Miałem podnieść precyzję na górnych 10% listy tak, aby kampania była opłacalna, nie zwiększając latencji batcha.

**Action:**
1. **Diagnoza przed zmianami**: analiza błędów wg segmentów, krzywe uczenia (wysoki wariancja czy obciążenie?), sprawdzenie leakage i kalibracji. Okazało się, że model słabo radził sobie z nowymi klientami (cold start) i że etykieta była zdefiniowana zbyt późno.
2. **Dane i cechy**: dodałem cechy behawioralne z okien czasowych (trendy użycia 7/30/90 dni), cechy interakcji z supportem; poprawiłem definicję etykiety.
3. **Rzetelna ewaluacja**: podział czasowy zamiast losowego, stratyfikacja, przedziały ufności na bootstrapie.
4. **Model**: strojenie hiperparametrów (Optuna), wagi klas, kalibracja izotoniczna; porównałem z prostą regresją logistyczną jako baseline.
5. Ablacje, aby wiedzieć, które zmiany faktycznie pomagają.

**Result:** AUC-PR wzrosło z 0,31 do 0,39 (przedział ufności nie zawierał starej wartości), precyzja w top-10% wzrosła o ok. jedną czwartą, a test A/B potwierdził wyższy uplift retencji. Nauczyłem się, że największy zysk dała poprawa **danych i definicji etykiety**, nie zmiana algorytmu.

*(Liczby są ilustracyjne - w rozmowie podaj własne, prawdziwe.)*

#### Czego szukają rekruterzy
- Metodyczna diagnoza (błędy, bias/variance, leakage) przed próbą losowych poprawek.
- Właściwy baseline, uczciwa ewaluacja i mierzalny efekt (najlepiej biznesowy).
- Priorytet: dane > cechy > model; świadomość kompromisów (koszt, latencja, interpretowalność).
- Własny wkład i wnioski (learnings), nie ogólniki.

**Źródła:**
- [Google - Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [scikit-learn - Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Andrew Ng - Machine Learning Yearning (deeplearning.ai)](https://info.deeplearning.ai/machine-learning-yearning-book)

---

<a id="q331"></a>
### 331. Jak podszedłbyś do projektu z ograniczoną ilością danych z etykietami?

**Odpowiedź:**

Zacznij od pytania, **czy etykiety da się tanio zdobyć**, a potem stopniowo dobieraj techniki - od najtańszych do najbardziej złożonych.

#### 1. Zrozum problem i ustal ewaluację
- Ile etykiet mamy, jaka jakość, ile nieoznaczonych danych, jaki jest koszt oznaczania?
- Zbuduj mały, **wysokiej jakości zbiór testowy** (nie do treningu) i reprezentatywny; przy małych danych stosuj cross-validation i raportuj przedziały ufności.
- Prosty **baseline** (reguły, regresja logistyczna, kNN).

#### 2. Wykorzystaj wiedzę z zewnątrz
- **Transfer learning / fine-tuning** modeli pretrenowanych (ResNet/ViT, BERT), często z zamrożonym backbone i małą głowicą (linear probe); PEFT (LoRA).
- **Few-shot / zero-shot prompting** LLM lub CLIP, gdy zadanie na to pozwala.
- **Embeddingi** + prosty klasyfikator.

#### 3. Wykorzystaj dane nieoznaczone
- **Self-supervised pretraining** (SimCLR, MAE, masked LM) na własnych danych.
- **Semi-supervised**: pseudo-labeling, consistency regularization (FixMatch).
- **Weak supervision** (Snorkel): funkcje etykietujące, reguły heurystyczne, LLM jako labeler + weryfikacja próbki przez człowieka.

#### 4. Zwiększ efektywność etykietowania
- **Active learning**: etykietuj przykłady o największej niepewności (uncertainty sampling) lub różnorodności.
- Jasne wytyczne dla annotatorów, pomiar zgodności (Cohen's kappa), przegląd trudnych przypadków.

#### 5. Zwiększ efektywne dane i ogranicz overfitting
- **Augmentacja** (obrazy: flip/crop/mixup; tekst: back-translation; dane syntetyczne, z kontrolą jakości).
- Regularyzacja, prostszy model, early stopping, dropout, ensemble.
- Uwaga na **leakage** i dysbalans klas (stratyfikacja, wagi).

#### 6. Wdrożenie
- Uruchom model z human-in-the-loop, zbieraj etykiety z produkcji i iteruj (data flywheel).

Kolejność praktyczna: baseline -> pretrained + fine-tuning -> augmentacja -> active learning/semi-supervised -> reguły/weak supervision.

**Źródła:**
- [Sohn et al. (2020): FixMatch](https://arxiv.org/abs/2001.07685)
- [Chen et al. (2020): SimCLR](https://arxiv.org/abs/2002.05709)
- [Ratner et al. (2016): Data Programming (Snorkel)](https://arxiv.org/abs/1605.07723)
- [Settles: Active Learning Literature Survey](https://burrsettles.com/pub/settles.activelearning.pdf)

---

<a id="q332"></a>
### 332. Co zrobisz, jeśli model działa dobrze w testach, ale słabo na produkcji?

**Odpowiedź:**

Zacznij od **diagnozy**, nie od retrenowania. Typowe przyczyny dzielą się na problemy z ewaluacją offline, różnice danych oraz problemy inżynieryjne.

#### Możliwe przyczyny
1. **Wyciek danych (leakage)** - cecha zawiera informację z przyszłości/etykiety, preprocessing fitowany na całości, duplikaty między train a test. Offline wyniki były zawyżone.
2. **Nieprawidłowy podział** - losowy split szeregów czasowych, ta sama jednostka (użytkownik) w train i test.
3. **Training-serving skew** - inny kod cech w treningu i serwowaniu, inne wersje bibliotek, inna obsługa braków, różne tokenizery, różne rozmiary/format obrazu.
4. **Dryf danych** - covariate shift, concept drift, sezonowość; produkcyjny rozkład różni się od treningowego (nowa populacja użytkowników).
5. **Niereprezentatywny zbiór testowy** - selekcja, nadreprezentacja "łatwych" przypadków.
6. **Nadmierne dopasowanie do zbioru walidacyjnego** przez wielokrotne strojenie.
7. **Metryka offline != metryka biznesowa**; pętle sprzężenia zwrotnego.
8. **Problemy operacyjne** - opóźnione/brakujące cechy, timeouty, kwantyzacja, błędy w pipeline.

#### Plan działania (priorytety)
1. **Potwierdź problem**: czy metryka produkcyjna jest poprawnie mierzona (etykiety opóźnione, wielkość próby)? Porównaj z wartością offline.
2. **Odtwórz predykcje**: zapisz wejścia produkcyjne (logging) i uruchom je offline - jeśli wyniki się różnią, to skew/błąd pipeline'u; jeśli takie same, to dryf/dane.
3. **Porównaj rozkłady** cech train vs prod (PSI, KS-test, monitoring dryfu), sprawdź braki i zakresy.
4. **Analiza błędów** wg segmentów (kraj, urządzenie, nowi użytkownicy).
5. **Audyt leakage** i procedury podziału; przygotuj nowy zbiór walidacyjny z okresu produkcyjnego.
6. **Naprawa**: jeden kod cech (feature store), retrening na świeżych danych, ważenie próbek, augmentacja pod domenę, kalibracja, fallback/reguły, wycofanie modelu (rollback).

#### Zapobieganie
- Testy A/B, **shadow deployment**, canary, monitoring danych i predykcji, alerty na dryf, regularny retrening, kontrola wersji danych i modeli.

**Źródła:**
- [Google - Rules of Machine Learning (training-serving skew)](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Concept drift (Wikipedia)](https://en.wikipedia.org/wiki/Concept_drift)
- [Evidently AI - Data drift docs](https://docs.evidentlyai.com/)
- [scikit-learn - Common pitfalls (data leakage)](https://scikit-learn.org/stable/common_pitfalls.html)

---

<a id="q333"></a>
### 333. Jak jesteś na bieżąco z postępami w ML?

**Odpowiedź:**

Pytanie sprawdza **samodyscyplinę, krytyczne myślenie i praktyczne podejście** - nie liczy się długa lista źródeł, lecz system oraz przykład zastosowania wiedzy.

#### Przykładowa odpowiedź
"Traktuję to jako stały, ograniczony czasowo proces (ok. 3-4 h tygodniowo):

- **Artykuły**: przeglądam arXiv (cs.LG, cs.CL, cs.CV) i newslettery/agregatory, wybieram 1-2 prace tygodniowo do dokładnego przeczytania; śledzę konferencje (NeurIPS, ICML, ICLR, ACL, CVPR). Czytam **abstrakt, wnioski i ablacje**, żeby ocenić, czy wynik jest wiarygodny i odtwarzalny.
- **Praktyka**: odtwarzam kluczowe wyniki lub uruchamiam kod (Hugging Face, PyTorch) na małym eksperymencie - to najlepszy filtr hype'u.
- **Społeczność**: blogi inżynieryjne firm, podcasty, dyskusje (grupy czytelnicze w pracy, meetupy), kod open-source, dokumentacja bibliotek i notatki z release'ów.
- **Kursy i książki** dla fundamentów (Stanford CS229/CS224n/CS231n, Deep Learning Book) - fundamentalna wiedza starzeje się wolniej niż narzędzia.
- **Dzielenie się**: prezentuję ciekawe prace zespołowi, prowadzę notatki/wiki.

Ostatnio np. zapoznałem się z [konkretna technika, np. LoRA/RAG/FlashAttention], zrobiłem mały proof of concept i na jego podstawie zaproponowałem usprawnienie w naszym projekcie (np. skrócenie treningu o X%)."

#### Na co zwracają uwagę rekruterzy
- Konkretne, aktualne przykłady (nazwij realne prace/narzędzia, które znasz).
- **Krytyczne podejście**: odróżniasz trwałe postępy od hype'u, sprawdzasz benchmarki, koszty i ograniczenia.
- Zdolność do przekładania nowości na wartość biznesową i zespołową.
- Równowaga: fundamenty + nowinki, a nie ślepe gonienie za każdym trendem.

**Źródła:**
- [arXiv - Machine Learning (cs.LG) listing](https://arxiv.org/list/cs.LG/recent)
- [Stanford AI Index Report](https://aiindex.stanford.edu/report/)
- [Hugging Face - Daily Papers](https://huggingface.co/papers)
- [Stanford CS229 - Machine Learning](https://cs229.stanford.edu/)

---

<a id="q334"></a>
### 334. Opowiedz o wymagającym projekcie ML, nad którym pracowałeś/aś. Jaki był cel? Jaka była Twoja rola? Z jakimi wyzwaniami się mierzyłeś/aś? Jak je pokonałeś/aś? Jaki był wynik? Czego się nauczyłeś/aś?

**Odpowiedź:**

Pytanie samo podaje szkielet odpowiedzi: **cel -> rola -> wyzwania -> działania -> wynik -> wnioski**. Wybierz projekt, który znasz w szczegółach technicznych i który miał mierzalny efekt; przygotuj wersję 2-3 minutowa i pogłębienia na pytania.

#### Przykładowa odpowiedź (szablon)
**Cel:** Zbudowanie systemu wykrywania anomalii w transakcjach, który ograniczy straty z oszustw bez nadmiernego blokowania uczciwych klientów (cel: recall >= 80% przy precyzji >= 30% na top alertach).

**Rola:** Byłem głównym inżynierem ML w zespole 5 osób: projekt cech, modelowanie, ewaluacja i wdrożenie (współpraca z data engineerem i analitykami fraud).

**Wyzwania i działania:**
- *Skrajny dysbalans klas (~0,2% fraudów) i opóźnione etykiety* - podział czasowy, wagi klas/focal loss, metryki PR-AUC i koszt biznesowy zamiast accuracy; okno "dojrzewania" etykiet.
- *Dryf i adaptacja atakujących* - monitoring dryfu (PSI), harmonogram retreningu, cechy odporne na zmiany.
- *Latencja < 50 ms* - uproszczenie modelu (gradient boosting zamiast dużej sieci), feature store, cache cech.
- *Wyjaśnialność dla analityków* - SHAP i powody alertów.
- *Wyciek danych w cechach agregowanych* - poprawka na obliczenia point-in-time.

**Wynik:** Recall 82% przy precyzji 35%, oszczędności rzędu X% strat (podaj realną liczbę), fałszywe alarmy spadły o Y%. Model działa w produkcji z monitoringiem.

**Wnioski:** Jakość danych i definicja etykiety ważniejsze niż architektura; proste modele + dobry pipeline wygrywają; wczesne uzgodnienie metryk biznesowych z interesariuszami; automatyzacja monitoringu od pierwszego dnia.

*(Liczby są ilustracyjne - używaj własnych, prawdziwych.)*

#### Wskazówki
- Mów "ja", nie tylko "my" - pokaż własny wkład, ale doceń zespół.
- Pokaż **trade-offy** i decyzje (dlaczego ten model, dlaczego ta metryka) oraz przyznaj się do błędów i tego, co zrobiłbyś inaczej.
- Łącz szczegóły techniczne z efektem biznesowym; przygotuj się na pytania: alternatywy, skalowanie, monitoring, etyka/prywatność.

**Źródła:**
- [Google - Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Amazon Leadership Principles (przykład kryteriów oceny behawioralnej)](https://www.amazon.jobs/en/principles)
- [Job interview (Wikipedia) - behavioral interview](https://en.wikipedia.org/wiki/Job_interview)

---

<a id="q335"></a>
### 335. Dokąd zmierzają ML/AI w ciągu najbliższych 5 lat?

**Odpowiedź:**

To pytanie o **perspektywę i osąd**, nie o wróżenie. Dobra odpowiedź: kilka trendów z uzasadnieniem, świadomość niepewności i ograniczeń oraz odniesienie do roli/firmy. Poniżej przykładowe tezy.

#### Trendy
1. **Modele foundation i agenci**: LLM/multimodalne modele stają się warstwą ogólnego przeznaczenia; rośnie znaczenie **agentów** (tool use, planowanie, MCP), RAG i systemów złożonych z wielu komponentów, nie pojedynczego modelu. Kluczowe będą niezawodność, ewaluacja i koszty.
2. **Efektywność**: mniejsze, wyspecjalizowane modele (destylacja, kwantyzacja, MoE, PEFT/LoRA), inferencja na urządzeniach brzegowych, optymalizacja kosztu tokenu i energii. Prawa skalowania (Kaplan i in. 2020) sugerują zyski ze skali, ale rośnie też rola jakości danych i reasoningu w czasie inferencji.
3. **Dane**: dane syntetyczne, kuracja, prywatność (federated learning, differential privacy), licencje i prawa autorskie.
4. **Bezpieczeństwo, alignment i regulacje**: EU AI Act, ocena ryzyka, red-teaming, interpretowalność, wykrywanie halucynacji.
5. **MLOps -> LLMOps/AgentOps**: ewaluacja ciągła, obserwowalność, guardrails, governance.
6. **AI w nauce i przemyśle**: biologia (AlphaFold), materiały, medycyna, robotyka i modele "embodied", automatyzacja pracy wiedzy - z człowiekiem w pętli.
7. **Nowe architektury i paradygmaty**: alternatywy dla transformerów (SSM, np. Mamba), multimodalność, długi kontekst, pamięć.

#### Zastrzeżenia
- Prognozy w tej dziedzinie często się nie sprawdzają; trudno przewidzieć tempo. Warto to zaznaczyć.
- Ograniczenia: koszty obliczeń i energii, niezawodność, dostępność danych, kwestie prawne i społeczne.

#### Zakończenie w kierunku roli
Podkreśl, jak te trendy wpływają na Twoje umiejętności (ewaluacja, wdrożenia, praca z LLM, odpowiedzialne AI) i co chcesz rozwijać w tej firmie.

**Źródła:**
- [Kaplan et al. (2020): Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Bommasani et al. (2021): On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258)
- [Stanford AI Index Report](https://aiindex.stanford.edu/report/)

---

<a id="q336"></a>
### 336. Dlaczego interesuje Cię ta rola/firma?

**Odpowiedź:**

Rekruter sprawdza **motywację, dopasowanie i przygotowanie**. Odpowiedź powinna łączyć trzy elementy: (1) firma/produkt/problem, (2) rola i wymagane umiejętności, (3) Twój rozwój i wkład. Unikaj ogólników ("duża firma", "dobre zarobki").

#### Struktura odpowiedzi
1. **Co Cię przyciąga w problemie/produkcie** - konkretne, oparte na researchu (produkt, publikacje, blog inżynieryjny, skala danych, misja).
2. **Dlaczego ta rola do Ciebie pasuje** - Twoje doświadczenie wprost powiązane z wymaganiami z ogłoszenia.
3. **Co wniesiesz** - mierzalny przykład z przeszłości.
4. **Jak chcesz się rozwijać** - i jak to pasuje do ścieżki w firmie (kultura, mentoring, technologie).

#### Przykładowa odpowiedź
"Zależało mi na pracy tam, gdzie modele ML trafiają do realnych użytkowników na dużą skalę, a nie kończą jako prototypy. Śledzę wasz produkt rekomendacji i zainteresował mnie artykuł na waszym blogu o [konkretny temat, np. ewaluacja online i eksperymenty A/B] - to obszar, w którym sam zbudowałem [konkretny projekt] i podniosłem [metryka] o [wartość]. Ta rola łączy modelowanie z wdrożeniami i monitoringiem, a to dokładnie te umiejętności, które chcę pogłębiać, szczególnie [np. systemy rankingowe/LLM]. Podoba mi się też kultura eksperymentowania i otwartość na dzielenie się wiedzą, o której czytałem w opiniach i rozmowach z waszymi inżynierami."

#### Jak się przygotować
- Przeczytaj ogłoszenie, produkt, blog techniczny, publikacje, aktualności, stack technologiczny, wartości firmy.
- Miej 2-3 konkretne punkty styku i jedno pytanie zwrotne do rekrutera.
- Bądź szczery: dopasowanie ma być prawdziwe, bo rekruterzy wychwytują wyuczone formułki.
- Nie krytykuj obecnego pracodawcy; mów o tym, do czego zmierzasz, a nie od czego uciekasz.

**Źródła:**
- [Job interview (Wikipedia)](https://en.wikipedia.org/wiki/Job_interview)
- [Amazon Leadership Principles (przykład wartości firmy)](https://www.amazon.jobs/en/principles)
- [Google - Rules of Machine Learning (kontekst pracy inżyniera ML)](https://developers.google.com/machine-learning/guides/rules-of-ml)

---

<a id="q337"></a>
### 337. Opisz sytuację, w której Twój model zawiódł lub nie osiągnął oczekiwanych wyników. Co zrobiłeś/aś?

**Odpowiedź:**

Rekruter chce zobaczyć **odpowiedzialność, umiejętność uczenia się z błędów i dojrzałość inżynierską**. Wybierz prawdziwą porażkę o realnym (ale nie katastrofalnym) wpływie, nie "słabość, która jest siłą". Nie obwiniaj innych.

#### Przykładowa odpowiedź (STAR)
**Situation:** Wdrożyłem model prognozujący popyt, który offline miał MAPE ok. 8%, ale po dwóch tygodniach w produkcji błąd wynosił ok. 20%, co powodowało braki magazynowe.

**Task:** Szybko ustabilizować sytuację i znaleźć przyczynę.

**Action:**
1. **Natychmiast**: wróciłem do poprzedniego modelu (rollback) i poinformowałem interesariuszy o skali problemu i planie.
2. **Diagnoza**: porównanie rozkładów cech offline vs produkcja, odtworzenie predykcji; znalazłem dwie przyczyny: (a) cechy opóźnień sprzedaży były w treningu liczone z danych, które w produkcji docierały z opóźnieniem (**leakage/skew**), (b) losowy split zamiast czasowego zawyżył wyniki offline.
3. **Naprawa**: point-in-time feature engineering, walidacja **walk-forward**, wspólny kod cech dla treningu i serwowania, testy danych.
4. **Zapobieganie**: shadow deployment przed każdym wdrożeniem, monitoring błędu i dryfu z alertami, checklista przed wdrożeniem, post-mortem bez szukania winnych.

**Result:** Po poprawkach produkcyjny MAPE spadł do ok. 10%, zgodnie z oczekiwaniami; proces (shadow mode + checklista) został przyjęty przez cały zespół.

**Learning:** Wynik offline jest hipotezą, a nie faktem; wiarygodna ewaluacja i obserwowalność są równie ważne jak sam model.

*(Liczby są ilustracyjne - użyj własnej historii.)*

#### Na co zwracają uwagę rekruterzy
- Przyjęcie odpowiedzialności, przejrzysta komunikacja, szybka mitygacja.
- Systematyczna analiza przyczyn źródłowych (root cause) i trwałe usprawnienia procesu.
- Brak defensywności, konkretne wnioski i ich zastosowanie w kolejnych projektach.

**Źródła:**
- [Google - Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [scikit-learn - Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)
- [Concept drift (Wikipedia)](https://en.wikipedia.org/wiki/Concept_drift)

---

<a id="q338"></a>
### 338. Jak postąpisz w przypadku nieporozumień z kolegami dotyczących wyboru modeli lub podejść?

**Odpowiedź:**

Rekruter ocenia **komunikację, dojrzałość zespołową i orientację na dowody**. Dobra odpowiedź: spór o model rozstrzygasz **danymi i wspólnymi kryteriami**, nie stanowiskiem czy hierarchią, i dbasz o relacje.

#### Podejście krok po kroku
1. **Zrozum perspektywę drugiej strony**: zadaj pytania, sparafrazuj argumenty ("Czy dobrze rozumiem, że zależy Ci na interpretowalności?"). Często różnica wynika z innych priorytetów (latencja, koszt, utrzymanie).
2. **Ustalcie wspólny cel i kryteria sukcesu**: metryka biznesowa i techniczna, ograniczenia (latencja, koszt, interpretowalność, czas wdrożenia, ryzyko).
3. **Rozstrzygnij eksperymentem**: szybkie porównanie na wspólnym zbiorze i protokole ewaluacji (te same foldy, ten sam split czasowy), z przedziałami ufności i testami istotności; prosty baseline jako punkt odniesienia; ograniczony czasowo proof of concept.
4. **Uwzględnij szerszy koszt**: złożoność utrzymania, monitoring, wyjaśnialność, skalowalność - nie tylko 1% różnicy metryki.
5. **Zdecyduj i zaangażuj się (disagree and commit)**: jeśli brak jednoznacznych danych, zdecyduj według ustalonego procesu (lub eskaluj do tech leada), udokumentuj decyzję (design doc/ADR), zaplanuj punkt weryfikacji.
6. **Zachowaj profesjonalizm**: krytykuj pomysły, nie osoby; przyznaj rację, gdy dane wskażą inaczej.

#### Przykładowa odpowiedź (STAR, skrót)
"Kolega chciał wdrożyć głęboką sieć, ja proponowałem gradient boosting. Zamiast dyskutować, uzgodniliśmy wspólny protokół walidacji i metryki (AUC-PR i latencja p95). Zrobiliśmy dwutygodniowe porównanie: sieć dała +0,5 p.p. AUC-PR, ale 6x wyższy koszt inferencji i gorszą wyjaśnialność. Przedstawiliśmy wyniki zespołowi i zdecydowaliśmy się na boosting z planem ponownej oceny sieci, gdy urośnie zbiór danych. Współpraca się umocniła, a proces porównań trafił do naszych wytycznych."

#### Czego unikać
- Wygrywania za wszelką cenę, ataków personalnych, forsowania własnej wersji bez danych.
- Ignorowania perspektywy zespołu; przeciągania decyzji bez końca.

**Źródła:**
- [Google - Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Conflict resolution (Wikipedia)](https://en.wikipedia.org/wiki/Conflict_resolution)
- [Demšar (2006): Statistical Comparisons of Classifiers over Multiple Data Sets (JMLR)](https://jmlr.org/papers/v7/demsar06a.html)

---
