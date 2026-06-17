# Preparação para a banca — Apresentação CMP617

**Trabalho:** Avaliação Semântica de Super-Resolução Temporal em Imagens Sentinel-2 por Métricas de PLN
**Autor:** Marcel Fernandes Gomes — PPGC/UFRGS

> Material de defesa baseado **apenas na apresentação** (`apresentacao.pdf`, 14 slides). Inclui: pitch, números-chave para decorar, banco de perguntas & respostas estilo banca, perguntas-pegadinha e glossário de siglas.

---

## 1. Pitch de 30 segundos (abertura / fechamento)

> "Super-resolução recupera detalhe em imagens de satélite, mas as métricas clássicas — PSNR e SSIM — só medem pixel, não significado. Eu proponho um pipeline que usa modelos de visão-linguagem e métricas de PLN para medir se o detalhe recuperado é **semanticamente real**. A contribuição central é uma avaliação cruzada que converte as respostas do GeoChat em estimativa de cobertura do solo e compara com um ground truth vetorial, gerando um erro interpretável em pontos percentuais. Validei em duas regiões distintas (953 tiles): a super-resolução reduz o erro de cobertura de 22,7 para 19,1 pp, com ganhos maiores em solo exposto e agricultura."

**Frase-âncora:** *"Eu não avalio a super-resolução pelos pixels; eu avalio pelo que ela permite reconhecer."*

---

## 2. Números-chave para memorizar

| Item | Valor |
|---|---|
| Resolução LR / SR / GT raster | ≈11,5 / ≈2,4 / ≈1 m/px |
| SR = fusão temporal de | 8 observações |
| Tile | 64×64 px no espaço LR (~0,73 km²) |
| Regiões | R1 = 575 tiles (mista, água frequente) · R2 = 378 (agrícola, água rara) · **pooled = 953** |
| Distância entre regiões | ~110 km |
| **RemoteCLIP img-img Δ** | **+0,508 → SR>LR em 952/953** (R1 +0,505 · R2 +0,512) |
| CLIP genérico img-img Δ | +0,352 → 930/953 |
| BLIP TF-IDF (Cenário B) Δ | +0,201 → 685/953 |
| Alucinações BLIP na LR | 20% → **11%** com SR |
| VQA L1/L2/L3 (TF-IDF Δ) | +0,228 / +0,131 / +0,122 |
| **VQA×GeoPackage MAE global** | **LR 22,7 → SR 19,1 pp (Δ +3,6)** |
| Hierarquia (MAE global pooled) | **GT 17,4 < SR 19,1 < LR 22,7** |
| SR fecha da lacuna LR→GT | **~68%** (3,6 de 5,3 pp) |
| Maiores ganhos (pooled) | **solo exposto +11,1** · agricultura +6,3 |
| Achado de falha | **água: R1 +1,9 / R2 −8,7 / pooled −2,3** |
| Significância | teste de sinais binomial, **p < 0,001** em todas as métricas |
| Cenário C (RemoteCLIP) | gap cai +0,508 → +0,421 |

---

## 3. Banco de perguntas & respostas

### A. Motivação e problema

**Q1. Por que PSNR e SSIM não bastam para avaliar super-resolução?**
Porque operam em nível de pixel e medem fidelidade fotométrica, não significado. Uma imagem SR pode ter PSNR alto e ainda introduzir texturas plausíveis porém incorretas. Existe o *perception-distortion tradeoff*: melhorar a qualidade perceptual frequentemente piora a fidelidade de pixel. Eu quero medir se o conteúdo recuperado é **semanticamente coerente** com o terreno, e isso o PSNR/SSIM não capturam.

**Q2. O que você quer dizer com "semanticamente correto"?**
Que o conteúdo recuperado corresponde às classes e feições reais do terreno — floresta, água, solo exposto, áreas agrícolas — e não a texturas inventadas. Operacionalmente: se um modelo de visão-linguagem descreve a imagem SR, a descrição bate melhor com a realidade (ortofoto e cadastro vetorial) do que a descrição da imagem LR.

**Q3. Por que usar PLN/linguagem para avaliar imagem? Não é indireto?**
A linguagem é uma camada semântica interpretável: ao fazer o modelo *descrever* ou *responder perguntas* sobre a imagem, eu acesso o que é reconhecível nela. Isso tem duas vantagens sobre métricas de pixel: (i) não exijo um par de alta resolução perfeitamente alinhado; (ii) obtenho uma medida interpretável por humanos. É a mesma lógica de avaliação por referência textual usada em legendagem de imagens.

**Q4. Qual a pergunta de pesquisa?**
As imagens Sentinel-2 super-resolvidas por fusão temporal recuperam conteúdo semântico **mensurável** frente à baixa resolução, quando avaliadas por métricas de PLN e modelos visão-linguagem? Resposta: sim, e eu quantifico esse ganho sem usar métricas de pixel.

### B. Dados e metodologia

**Q5. O que é super-resolução temporal? Como a fusão de 8 observações funciona?**
O satélite revisita a mesma área periodicamente; cada passagem tem pequenos deslocamentos sub-pixel. Esses deslocamentos carregam informação de alta frequência complementar. Um modelo combina (funde) as 8 observações de baixa resolução para reconstruir detalhe abaixo da resolução nativa — é *multi-image super-resolution*. O produto "blended" resultante tem ≈2,4 m/px.

**Q6. Você produziu a super-resolução?**
Não — a SR é o **insumo** do meu trabalho; minha contribuição é o **pipeline de avaliação semântica**, não o método de SR. Isso é importante: eu avalio um produto de SR, não proponho um novo super-resolvedor.

**Q7. De onde vêm os dados?**
Cenas Sentinel-2 (canal TCI, composição RGB das bandas 4-3-2) da Agência Espacial Europeia, no Sul do Brasil. A referência raster é uma **ortofoto aérea** (~1 m/px) e a referência vetorial é um **GeoPackage** com 7 camadas de cobertura do solo de anotação manual.

**Q8. Por que a LR tem 11,5 m/px se o Sentinel-2 nativo é 10 m/px?**
A resolução nativa das bandas de cor é 10 m/px. Ao reprojetar tudo para EPSG:3857 (Web Mercator), há distorção de escala que depende da latitude (fator ≈1/cos φ); no Sul do Brasil isso eleva o pixel efetivo para ≈11,5 m. É um efeito conhecido do Web Mercator.

**Q9. O que é EPSG:3857 e por que reprojetar?**
É o código do Web Mercator, projeção usada por mapas web. Reprojetei todas as camadas (Sentinel-2, ortofoto, vetores) para o mesmo sistema de coordenadas para que se alinhem espacialmente. Limitação assumida: o Web Mercator distorce escala (daí o 11,5 m); uma projeção equivalente em área seria mais rigorosa para medir áreas.

**Q10. Como foi feito o tiling? Por que 64×64?**
A imagem é particionada em uma grade de tiles de 64×64 px no espaço LR (~0,73 km²/tile); tiles com >50% de NoData são descartados. 64×64 é um compromisso: grande o bastante para conter contexto de cena, pequeno o bastante para ter cobertura razoavelmente homogênea e gerar muitos tiles para estatística.

**Q11. Como você atribui a classe de cada tile?**
Interseciono os polígonos vetoriais com a *bounding box* do tile e atribuo a **classe dominante** quando a interseção supera 5% da área do tile. Na R1, 437 dos 575 tiles têm classe dominante; na R2, todos os 378.

**Q12. O que são os Cenários A, B e C e por que três?**
São três escolhas de *referência* para comparar as descrições:
- **A — GT Vetorial:** referência = nome da classe em inglês.
- **B — GT Raster (1 m):** referência = legenda/embedding da ortofoto.
- **C — GT Raster reamostrado (~2,4 m):** igual ao B, mas com a ortofoto degradada para a resolução da SR — isola a penalidade de resolução diferencial.
Os três isolam fatores diferentes: A testa correspondência léxica direta, B usa a melhor referência disponível, C controla o efeito de resolução.

**Q13. Por que R1 tem 575 e R2 tem 378 tiles?**
São extensões geográficas e quantidades de área válida diferentes. Não é duplicata — R2 é outra paisagem (mais agrícola, água rara), ~110 km distante. Os esquemas de dados são idênticos para permitir consolidação (pooled = 953).

### C. Modelos

**Q14. O que é o BLIP?**
*Bootstrapping Language-Image Pre-training* — um modelo de legendagem pré-treinado em pares imagem-texto da Web. Gera descrições de vocabulário diverso, porém **genérico**, não especializado em sensoriamento remoto.

**Q15. O que é o GeoChat e por que é melhor aqui?**
É um modelo de visão-linguagem com arquitetura LLaVA (encoder visual CLIP + decodificador LLaMA-7B), **fine-tuned em 318 mil pares de instrução geoespaciais** (legendagem, localização e VQA sobre imagens de satélite). Por ter visto o domínio, descreve melhor cenas de sensoriamento remoto e não colapsa na baixa resolução como o BLIP.

**Q16. O que é CLIP? E RemoteCLIP?**
CLIP aprende, por *contrastive learning*, um espaço latente compartilhado entre imagem e texto; a similaridade cosseno nesse espaço mede alinhamento semântico. RemoteCLIP é o CLIP **especializado** para sensoriamento remoto (datasets RSICD, RSITMD, UCM), com representações mais discriminativas para classes de cobertura do solo.

**Q17. Por que escolheu GeoChat e não outro VLM?**
Por ser um VLM aberto e especializado em sensoriamento remoto, cobrindo legendagem e VQA — exatamente as tarefas de que preciso. Assumo como limitação que o comportamento de saturação observado pode ser específico desse modelo (ver Q35).

### D. Métricas de PLN (a banca tende a aprofundar aqui)

**Q18. O que é BLEU-1 e como calcula?**
BLEU mede sobreposição de n-gramas entre texto gerado e referência, orientada à **precisão**. BLEU-1 usa apenas unigramas: a fração de palavras do texto gerado que aparecem na referência. É uma métrica **léxica** (correspondência exata de palavras). Usei suavização para evitar zeros em textos curtos.

**Q19. O que é ROUGE-1 e qual a diferença para o BLEU?**
ROUGE-1 também conta unigramas em comum, mas é orientada à **revocação** e reportada como **F1** (combina precisão e revocação). BLEU pune o que o gerado tem a mais; ROUGE valoriza o que ele cobre da referência. As duas são léxicas e complementares.

**Q20. O que é TF-IDF e como vira similaridade?**
Cada texto vira um vetor onde cada dimensão é uma palavra ponderada por TF-IDF (frequência do termo × raridade no corpus, via IDF). A similaridade é o **cosseno** entre os dois vetores. O IDF reduz o peso de palavras muito comuns, destacando termos discriminativos. **Mas é léxica**: "forest" e "woodland" são dimensões independentes — não captura sinonímia.

**Q21. O que é similaridade por cosseno?**
O cosseno do ângulo entre dois vetores: 1 = mesma direção (muito similares), 0 = ortogonais. Com TF-IDF (valores ≥0) fica em [0,1]. Vantagem: normaliza pelo tamanho dos vetores, comparando textos longos e curtos de forma justa. É o modelo vetorial clássico de recuperação de informação (Salton).

**Q22. O que é o CLIPScore? Por que o fator 2,5 e o max(·,0)?**
CLIPScore = w · max(cos(e_texto, e_imagem), 0), com w = 2,5. É a similaridade cosseno entre embeddings, reescalada por 2,5 (as similaridades do CLIP costumam ficar numa faixa estreita; o fator espalha os valores) e truncada em zero (similaridades negativas são raras e sem sentido prático). Uso-o como métrica **comparativa** entre resoluções, então a magnitude absoluta não importa — importa o Δ entre LR e SR.

**Q23. Qual a diferença entre CLIPScore imagem-texto e imagem-imagem?**
No modo imagem-texto, comparo o embedding de uma legenda com o de uma imagem. No modo **imagem-imagem**, extraio embeddings visuais de LR, SR e da ortofoto direto pelo encoder CLIP e comparo imagem com imagem — **sem depender de modelo de legendagem**. Foi o resultado mais limpo e estável, porque elimina o ruído da geração de texto.

**Q24. O que é o all-MiniLM-L6-v2 / a "similaridade semântica"?**
É um modelo **Sentence-BERT (SBERT)**: uma rede siamesa baseada em BERT que projeta sentenças inteiras num espaço vetorial onde o cosseno reflete proximidade de **significado**, não de palavras. Diferente do TF-IDF, captura paráfrase e sinonímia. Usei-o para comparar as respostas do VQA semanticamente.

**Q25. Por que os scores léxicos ficam quase zero no Cenário A?**
Porque a referência é o nome da classe ("forest") e o BLIP raramente reproduz esse termo literalmente — descreve a cena com outras palavras. Como BLEU/ROUGE/TF-IDF são léxicas, sobreposição quase nula → score ≈0. **Isso é a limitação de sinonímia do TF-IDF acontecendo na prática** — e é justamente o que motiva migrar para embeddings e para a avaliação cruzada quantitativa.

**Q26. O que é MAE e por que pontos percentuais?**
MAE = erro absoluto médio. Na avaliação cruzada, comparo a porcentagem de cobertura estimada (pelo GeoChat) com a porcentagem real (do GeoPackage), por classe, e tiro a média dos erros absolutos. A unidade é **ponto percentual (pp)** porque estou medindo diferença entre porcentagens — uma métrica diretamente interpretável.

### E. Resultados e interpretação

**Q27. Por que o RemoteCLIP é o resultado mais expressivo e estável?**
Porque é especializado no domínio (sensoriamento remoto) e mede direto, imagem contra imagem, sem o ruído da legendagem. Classifica a SR como mais próxima da ortofoto em **952/953 tiles**, com ganho quase idêntico nas duas regiões (+0,505 e +0,512) — daí "estável".

**Q28. O que significa "SR>LR em 952/953"?**
Que, tile a tile, a SR ficou mais parecida com a referência de alta resolução do que a LR em 952 dos 953 casos. É uma evidência pareada fortíssima; o teste de sinais binomial dá p < 0,001 (praticamente impossível por acaso).

**Q29. No Cenário C o gap cai de +0,508 para +0,421. O que isso prova?**
Que **parte** da vantagem medida era penalidade de resolução (a ortofoto a 1 m favorece naturalmente a SR a 2,4 m). Ao degradar a ortofoto para 2,4 m, removo esse efeito — e o ganho persiste (+0,421). Ou seja: a hierarquia LR < SR é **real**, não artefato de resolução diferencial.

**Q30. Se o GeoChat tem gap menor que o BLIP, isso não é um resultado pior?**
Não — é um resultado mais **honesto**. O BLIP inflava o gap porque colapsava (alucinava) na LR. O GeoChat descreve bem as duas resoluções e não alucina, então o gap menor mostra que o ganho real da SR é mais modesto do que o BLIP sugeria. A queda do gap valida que parte do "ganho" do BLIP era artefato.

**Q31. O que são as alucinações do BLIP e por que acontecem?**
Em 20% dos tiles LR da R1, o BLIP gera descrições absurdas como "a blue sky with clouds" ou "a couple is kissing in the water". A resolução insuficiente não dá pistas suficientes, e o modelo recorre a cenas frequentes do seu pré-treinamento (Web) — o fenômeno de **alucinação** de modelos generativos. A SR reduz isso para 11%.

**Q32. Por que o nível L2 (estrutural) saturou?**
A pergunta q03 (contar campos agrícolas distintos) dá zero em **ambas** as resoluções, porque a escala do tile (~0,73 km²) e a resolução não permitem contar campos individuais de forma confiável. Foi uma pergunta inadequada à escala — eu reconheço e proponho reformulá-la.

**Q33. Por que a água piora em R2? (achado principal de falha)**
Na R2 a água é rara. A SR tende a **alucinar água** em tiles onde não há (gt_water = 0): a LR acerta estimando 0%, mas a SR estima ~25%, gerando erro grande. Por isso o MAE de água piora −8,7 pp na R2, enquanto melhora +1,9 pp na R1 (onde há água de verdade). É um modo de falha **invisível a um estudo de área única** — só a multi-região revelou.

**Q34. Por que os maiores ganhos são em solo exposto e agricultura?**
São classes com textura e padrão de média/alta frequência (sulcos de cultivo, manchas de solo) que a fusão temporal recupera bem. Solo exposto: +11,1 pp pooled (robusto nas duas regiões: +9,2 e +14,0). Agricultura: +6,3 pp pooled, especialmente forte na R2 agrícola (+9,2). São classes diretamente relevantes para monitoramento agrícola/ambiental.

**Q35. A hierarquia é GT < SR < LR. Por que a SR não alcança o GT?**
Porque a SR (2,4 m) ainda tem menos detalhe que a ortofoto (1 m): reconstrói conteúdo plausível, mas não recupera todo o detalhe real. Ainda assim, a SR **fecha ~68% da lacuna recuperável** (de 22,7 para 19,1, sendo 17,4 o piso do GT). Isso mostra ganho substancial sem alcançar o teto.

### F. Validade, estatística e limitações

**Q36. Como você sabe que o ganho é estatisticamente significativo?**
Apliquei um **teste de sinais binomial bicaudal** sobre os pares por tile (LR vs SR). Com 952/953 favoráveis, p < 0,001 — o resultado é incompatível com acaso. O mesmo vale para as demais métricas e para cada região isolada.

**Q37. Duas regiões permitem generalizar?**
Eu sou cuidadoso na linguagem: afirmo **consistência entre as duas regiões testadas**, não generalização universal. Duas regiões não estimam a variância entre paisagens — por isso ampliar o número de regiões é trabalho futuro. O próprio achado da água mostra que a generalização tem exceções dependentes da paisagem.

**Q38. O ganho é da super-resolução espacial ou da fusão temporal?**
Confundimento assumido como limitação. A SR funde 8 observações temporais; parte do ganho pode vir de redução de ruído pela média temporal, não só do aumento espacial. Meu pipeline mede a qualidade semântica do **produto final**, sem desacoplar os dois efeitos.

**Q39. O tile de 64×64 mistura classes. Isso não compromete o rótulo?**
Sim, é uma aproximação — por isso uso a **classe dominante** (>5%) e, na avaliação cruzada, comparo **porcentagens** de cobertura em vez de rótulo único, o que tolera mistura. Ainda assim, a heterogeneidade do tile é uma fonte de ruído reconhecida.

**Q40. O GeoChat estima porcentagens. Como confiar nisso?**
Eu **não** confio na exatidão absoluta dele — uso-o **comparativamente**. Mesmo modelo, mesma pergunta, três entradas (LR, SR, GT). O viés do modelo cancela, e o que sobra é o sinal LR vs SR. A prova disso é o MAE do próprio GT raster (17,4 pp): mesmo na melhor imagem, o GeoChat erra ~17 pp — esse é o piso de erro do modelo, não da resolução.

**Q41. Então por que o MAE do GT raster não é zero, se é o ground truth da pergunta?**
Ótima pergunta. O GT raster aqui é a *imagem* de referência (ortofoto), e o GeoChat ainda erra ao estimar porcentagens nela; além disso há ruído de anotação entre vetor e ortofoto e mistura de classes no tile. Os 17,4 pp isolam o erro **intrínseco do modelo + referência**, separado do erro de **resolução** (a faixa 22,7 → 17,4).

**Q42. Você validou as respostas manualmente?**
A validação é o próprio **GeoPackage anotado por humanos**: a avaliação cruzada confronta as estimativas com esse cadastro manual. Além disso, inspecionei tiles representativos (a figura de grid LR/SR/GT), incluindo o caso de alucinação de água da R2.

**Q43. Por que 5 classes no VQA×GeoPackage e 7 no GeoPackage?**
Agreguei para casar com o vocabulário do GeoChat e reduzir esparsidade: agricultura = campo + vegetação cultivada; solo exposto = terreno exposto + brejo/pântano; floresta, água e urbano permanecem.

### G. Contribuição e conexão com PLN

**Q44. Qual é a contribuição central e por que é nova?**
A **avaliação cruzada VQA × GeoPackage**: uso a resposta em linguagem natural do GeoChat (estimativa de cobertura) como predição e o cadastro vetorial como ground truth quantitativo, gerando um erro em pp. Isso **elimina a dependência léxica** das métricas de texto e não exige par de alta resolução alinhado. É a primeira métrica do pipeline que é ao mesmo tempo semântica e quantitativamente interpretável.

**Q45. Como o trabalho se conecta com a disciplina de PLN?**
Em três eixos do conteúdo: (i) **métricas de avaliação** — BLEU, ROUGE, similaridade por cosseno TF-IDF (modelo vetorial de RI); (ii) **similaridade semântica** — embeddings, SBERT, CLIPScore (cosseno em espaço de embeddings); (iii) **modelos multimodais / foundation models** — CLIP, BLIP, GeoChat. Há ainda o fenômeno de **alucinação** de modelos generativos, central no resultado do BLIP.

**Q46. A avaliação cruzada usa MAE, que é métrica de regressão. Isso ainda é PLN?**
Sim, porque o **preditor é a saída em linguagem natural** de um modelo de visão-linguagem respondendo a uma pergunta em linguagem natural. A contribuição é metodológica dentro de avaliação de PLN: demonstro a limitação das métricas léxicas (sobreposição de palavras) e proponho ancorar a avaliação em uma referência quantitativa. O MAE é o instrumento; o objeto é a qualidade semântica da descrição.

**Q47. Onde esse trabalho se encaixa na "evolução do PLN"?**
Meu pipeline reproduz o próprio arco da disciplina: começa **léxico** (BLEU/ROUGE/TF-IDF — era do BoW), passa para **semântico** (SBERT/CLIPScore — era das embeddings/contextuais) e termina **ancorado** na realidade (MAE vs ground truth vetorial). A falha do Cenário A é literalmente a limitação de sinonímia do BoW/TF-IDF que justifica embeddings.

### H. Perguntas-pegadinha (esteja pronto)

**Q48. A super-resolução "inventa" detalhe. Isso não é perigoso para aplicações reais?**
Sim, e é exatamente por isso que o trabalho importa: métricas de pixel não detectam invenção plausível, mas a avaliação semântica detecta — o achado da **alucinação de água na R2** é prova disso. A mensagem prática é calibrar expectativas por tipo de paisagem antes de usar SR em decisão automática.

**Q49. Por que não usou métricas de QA padrão (exact match, F1 de tokens)?**
Porque as respostas do GeoChat são livres e numéricas (porcentagens), não rótulos fechados; exact match seria quase sempre zero. O MAE em pp captura melhor a proximidade da estimativa. Para as perguntas descritivas, usei TF-IDF e similaridade semântica, que toleram variação de redação.

**Q50. Por que TF-IDF e similaridade semântica dão Δ diferentes no VQA (léxico maior que semântico)?**
Porque o TF-IDF reage à mudança de **palavras** na superfície (a SR muda bastante a redação), enquanto o SBERT reage à mudança de **significado**. Ambos positivos → a SR melhora os dois; mas o Δ semântico menor indica que boa parte do ganho léxico é reformulação, não mudança de conteúdo. É uma leitura honesta da força do efeito.

**Q51. Qual é a maior fragilidade do seu trabalho?**
Duas: (1) um único VLM (GeoChat), então a saturação pode ser específica dele; (2) o confundimento SR-espacial vs fusão-temporal. Ambas são endereçáveis com mais modelos e um ablation, que coloco como trabalho futuro.

**Q52. Se tivesse mais tempo e recursos, o que faria?**
(i) Correlacionar as métricas semânticas com **tarefas downstream** reais (classificação/detecção), para mostrar utilidade prática; (ii) reformular as perguntas VQA de nível estrutural que saturaram; (iii) ampliar para mais regiões, datas e condições de nuvem, quantificando estatisticamente a variabilidade entre paisagens.

---

## 4. Glossário de siglas

| Sigla | Significado | No trabalho |
|---|---|---|
| **PLN / NLP** | Processamento de Linguagem Natural / *Natural Language Processing* | Disciplina e arcabouço das métricas |
| **SR** | Super-Resolução (*Super-Resolution*) | Imagem reconstruída ≈2,4 m/px |
| **LR** | Baixa Resolução (*Low Resolution*) | Sentinel-2 nativo ≈11,5 m/px |
| **GT** | *Ground Truth* (verdade de referência) | Ortofoto (raster) e GeoPackage (vetorial) |
| **m/px** | metros por pixel | Resolução espacial |
| **pp** | pontos percentuais | Unidade do MAE de cobertura |
| **VLM** | *Vision-Language Model* (modelo visão-linguagem) | BLIP, GeoChat |
| **VQA** | *Visual Question Answering* | Eixo 3: 7 perguntas em 3 níveis |
| **BLIP** | *Bootstrapping Language-Image Pre-training* | Legendador genérico |
| **GeoChat** | VLM *grounded* para sensoriamento remoto | Legendagem + VQA especializados |
| **LLaVA** | *Large Language and Vision Assistant* | Arquitetura do GeoChat |
| **LLaMA** | *Large Language Model Meta AI* | Decodificador (7B) do GeoChat |
| **CLIP** | *Contrastive Language-Image Pre-training* | Embeddings imagem-texto |
| **RemoteCLIP** | CLIP especializado em sensoriamento remoto | Métrica imagem-imagem mais sensível |
| **CLIPScore** | Métrica de similaridade referência-livre baseada em CLIP | w·max(cos,0), w=2,5 |
| **BLEU** | *Bilingual Evaluation Understudy* | Métrica léxica (precisão de unigramas) |
| **ROUGE** | *Recall-Oriented Understudy for Gisting Evaluation* | Métrica léxica (F1 de unigramas) |
| **TF-IDF** | *Term Frequency–Inverse Document Frequency* | Vetor léxico + cosseno |
| **BERT** | *Bidirectional Encoder Representations from Transformers* | Base do SBERT |
| **SBERT** | *Sentence-BERT* | Similaridade semântica (all-MiniLM-L6-v2) |
| **MiniLM** | Modelo de linguagem compacto (all-MiniLM-L6-v2) | Embeddings de sentença |
| **MAE** | *Mean Absolute Error* (Erro Absoluto Médio) | Avaliação cruzada VQA×GeoPackage |
| **PSNR** | *Peak Signal-to-Noise Ratio* | Métrica de pixel (criticada) |
| **SSIM** | *Structural Similarity Index* | Métrica de pixel (criticada) |
| **GeoPackage** | Formato vetorial geoespacial aberto (.gpkg) | GT vetorial, 7 classes |
| **TCI** | *True Color Image* | Produto RGB do Sentinel-2 |
| **RGB / RGBA** | Red-Green-Blue (+ Alpha) | Composição de cor / ortofoto |
| **EPSG:3857** | Código da projeção Web Mercator | Sistema de coordenadas comum |
| **CRS** | *Coordinate Reference System* | Sistema de referência espacial |
| **ESA** | *European Space Agency* (Agência Espacial Europeia) | Operadora do Sentinel-2 |
| **L1 / L2 / L3** | Níveis da VQA: Semântico / Estrutural / Detalhe fino | Estratificação das 7 perguntas |
| **R1 / R2** | Região 1 / Região 2 | Áreas de validação (575 / 378 tiles) |
| **ViT** | *Vision Transformer* | Encoder do CLIP (clip-vit-base-patch32) |
| **RSICD / RSITMD / UCM** | Datasets de legendagem de sensoriamento remoto | Treino do RemoteCLIP |
| **FP16 / NF4** | Ponto flutuante 16-bit / *Normal Float* 4-bit | Precisão de execução do GeoChat |
| **V100** | GPU NVIDIA Tesla V100 | Hardware dos experimentos |
| **NLTK** | *Natural Language Toolkit* | Cálculo de BLEU/ROUGE |
| **STS / NLI** | *Semantic Textual Similarity* / *Natural Language Inference* | Tarefas que treinam SBERT |

---

*Dica final: se travar numa pergunta, volte à frase-âncora ("avalio pelo que a imagem permite reconhecer") e ao número de manchete (952/953 e 22,7→19,1 pp). Eles reconduzem qualquer resposta ao núcleo do trabalho.*

---

## 5. Simulação de banca — perguntas e avaliação de respostas

> Sessão realizada em 2026-06-10. Para cada pergunta: resposta dada, nota (0–10) e o que faltou / resposta completa.

---

### P1 — Motivação: PSNR/SSIM vs métricas de PLN

**Pergunta:** PSNR e SSIM são métricas consolidadas na literatura de super-resolução. Por que descartá-las em favor de métricas de PLN? Não seria mais rigoroso usar as duas em conjunto?

**Resposta dada:** As métricas PLN são complementares ao PSNR/SSIM, não substitutas. A vantagem é não precisar de alinhamento perfeito e ter uma métrica que avalia semântica, não apenas valor numérico de pixel.

**Nota: 6/10**

**O que faltou:**
- O argumento mais forte: o ***perception-distortion tradeoff*** — métodos de SR que maximizam qualidade perceptual tendem a ter PSNR baixo; o PSNR pode *penalizar* uma imagem visualmente melhor. Isso é o caso canônico que justifica métricas semânticas.
- No experimento, o GT é uma ortofoto de sensor diferente — PSNR/SSIM com referência de sensor diferente introduz ruído radiométrico e de alinhamento que vai além do problema de alinhamento geométrico.

---

### P2 — Dados: heterogeneidade dos tiles

**Pergunta:** Tiles de 64×64 px (~0,73 km²) frequentemente contêm mistura de classes. Como você lida com essa heterogeneidade na atribuição de rótulo e o quanto isso compromete os resultados?

**Resposta dada:** Faz-se interseção entre o tile e o GeoPackage para encontrar a classe dominante. O tamanho do tile é grande o suficiente para ser interpretável e pequeno o suficiente para gerar muitos tiles.

**Nota: 5/10**

**O que faltou:**
- Mencionar o **threshold de 5%** de interseção para considerar classe dominante — e justificar a escolha.
- Não respondeu "o quanto compromete os resultados": no Cenário A (rótulo único), a heterogeneidade é ruído real — por isso 138/575 tiles ficam sem classe e os scores são fracos.
- O ponto mais importante: a **avaliação cruzada VQA × GeoPackage resolve isso por design** — em vez de rótulo único, compara *porcentagens* de cobertura, tolerando heterogeneidade diretamente.

---

### P3 — TF-IDF no Cenário A

**Pergunta:** No Cenário A a referência é apenas o nome da classe (ex.: "forest"). Scores próximos de zero invalidam o TF-IDF nesse contexto? O que esse resultado realmente nos diz?

**Resposta dada:** É basicamente invalidado mas ainda aproveitável. TF-IDF não mede sinonímia — se a classe é "field" e a legenda diz "agriculture", o score é zero. Ainda assim motiva avançar para outras abordagens.

**Nota: 5/10**

**O que faltou:**
- "Ainda haverá correlação" ficou vago — correlação entre o quê?
- O score próximo de zero **não invalida o pipeline** — é informativo: mesmo com scores absolutos ínfimos, a comparação LR vs SR revela que a LR tem mais alucinações (20% → 11%). O discriminador funciona mesmo quando o valor absoluto é baixo.
- A narrativa mais forte: o Cenário A reproduz na prática a **limitação clássica do Bag-of-Words** que historicamente justificou o surgimento de embeddings. O pipeline recapitula o arco do PLN: léxico → semântico → ancorado em GT.
- Evitar "invalidado mas aproveitável" — é contraditório. Reformular como: resultado negativo é diagnóstico e motiva os Cenários B/C e a avaliação cruzada.

---

### P4 — RemoteCLIP: viés de resolução

**Pergunta:** RemoteCLIP foi treinado em imagens de alta resolução — isso não favoreceria sistematicamente a SR sobre a LR independentemente do conteúdo semântico?

**Resposta dada:** Não soube responder.

**Resposta completa:**
A objeção é legítima e deve ser reconhecida. O **Cenário C** foi desenhado para controlá-la: ao degradar a ortofoto para a resolução da SR (~2,4 m/px), remove-se a vantagem diferencial de resolução. O gap cai de +0,508 para +0,421 — confirmando que parte do ganho é penalidade de resolução diferencial, mas a hierarquia LR < SR persiste. Há sinal real além do viés. A resposta correta admite a limitação e aponta o Cenário C como controle parcial.

---

### P5 — Achado de falha: água na Região 2

**Pergunta:** A SR piora a estimativa de água na R2 (−8,7 pp). Por que isso acontece mecanicamente e o que implica para uso em produção?

**Resposta dada:** Quando a feição de água é rara, a SR tende a superestimar o valor, gerando alucinações que precisam ser revisadas.

**Nota: 5/10**

**O que faltou:**
- Não explicou o **mecanismo**: em tiles com gt_water = 0, a LR acerta estimando ~0% (sinal ambíguo e fraco, o modelo não arrisca). A SR, ao reconstruir texturas de maior frequência, gera superfícies que *se parecem com água* — reflexo, textura lisa — onde não há água. É o produto SR inventando detalhe plausível porém incorreto, não alucinação do VLM.
- A **implicação prática** ficou vaga: se usado para monitoramento de corpos d'água ou detecção de cheias em regiões predominantemente secas, haverá **falsos positivos sistemáticos**. A mensagem operacional é calibrar expectativas por tipo de paisagem antes de usar SR em decisão automática.

---

### P6 — VQA × GeoPackage como contribuição de PLN

**Pergunta:** Um avaliador poderia argumentar que VQA × GeoPackage é essencialmente regressão com MAE — fora do escopo de PLN. Como defender que pertence a um trabalho de PLN?

**Resposta dada:** Não soube responder.

**Resposta completa:**
Três argumentos:
1. **O preditor é linguagem natural**: o GeoChat recebe pergunta em linguagem natural e devolve resposta em linguagem natural; extrair números dessa resposta requer PLN. Sem a saída textual do modelo, não existe métrica.
2. **Contribuição à metodologia de avaliação de PLN**: demonstra a limitação das métricas léxicas e propõe ancorar a saída semântica de um VLM a uma referência quantitativa — problema central em avaliação de modelos multimodais.
3. **Arco histórico**: a contribuição está no terceiro degrau da progressão léxico → semântico → ancorado em GT, que é onde a pesquisa de avaliação de PLN está hoje.

**Frase-chave:** *"O MAE é a régua; o que estou medindo é o quanto um modelo de linguagem entende a semântica de uma imagem de satélite."*

---

### P7 — Confundimento SR espacial vs fusão temporal

**Pergunta:** O produto SR resulta de fusão de 8 observações temporais. Como você separa o ganho de resolução espacial do ganho de redução de ruído temporal? Se não separa, o que isso implica?

**Resposta dada:** O trabalho não avalia o método de SR, avalia o produto SR já gerado.

**Nota: 3/10**

**O que faltou:**
- A resposta deflecte em vez de responder. A banca não pede que se proponha novo método — questiona se a **conclusão** é válida.
- Resposta correta: reconhecer como limitação real (explicitamente listada). As conclusões devem ser lidas como "o produto SR gerado por fusão temporal melhora a avaliação semântica frente à observação LR individual" — não como "SR espacial por si só melhora". Os dois efeitos estão confundidos.
- Em trabalhos futuros: um **ablation** comparando LR individual vs média temporal de 8 LRs (sem SR espacial) vs SR isolaria o efeito. Isso transforma a limitação em agenda de pesquisa.
- Deflectar uma limitação metodológica real sem reconhecê-la é interpretado negativamente pela banca.

---

### P8 — Gap menor do GeoChat: resultado pior ou melhor?

**Pergunta:** O GeoChat produz gap menor que o BLIP em todas as métricas NLP. Como transformar isso em argumento a favor do pipeline?

**Resposta dada:** Não soube responder.

**Resposta completa:**
O gap menor do GeoChat é mais **honesto**, não pior. O BLIP infla o gap porque colapsa na LR (20% de alucinações) — o score da LR é artificialmente baixo. O GeoChat descreve bem ambas as resoluções e não alucina, então o gap menor representa o ganho semântico real da SR. Isso **valida que parte do gap do BLIP era artefato**. Além disso, precisamente por ser robusto na LR, o GeoChat se torna o modelo certo para VQA — se alucinasse na LR, as respostas seriam inúteis. A progressão lógica do pipeline depende disso: BLIP gap grande → suspeita de artefato → GeoChat gap menor → confirma → GeoChat robusto → VQA confiável → contribuição quantitativa.

---

### P9 — Objeção à validação multi-região

**Pergunta:** Duas regiões no mesmo estado, mesmo sensor, mesmo método de SR — isso não é replicação, não validação. Como responder?

**Resposta dada:** Não soube responder.

**Resposta completa:**
Três argumentos:
1. **Reconhecer o que a objeção tem de certo**: duas regiões não permitem inferência estatística sobre variância entre todas as paisagens. As conclusões são explicitamente limitadas a consistência entre as duas regiões testadas.
2. **A objeção confunde o objeto de validação**: não se está validando um método de SR (que exigiria diversidade de sensor), mas um **pipeline de avaliação**. O eixo de variação relevante é diversidade de distribuição de cobertura do solo — e R1 (mista, água frequente) vs R2 (agrícola, água rara) são genuinamente distintas nesse eixo.
3. **O achado da água prova que as regiões são diferentes o suficiente**: se fosse replicação, a água se comportaria igual. A inversão (+1,9 / −8,7 pp) é prova de que as regiões expõem distribuições distintas — e que duas regiões já foram suficientes para revelar um modo de falha que um estudo single-area jamais detectaria.

---

### P10 — MAE da ortofoto ≠ 0

**Pergunta:** Quando o GeoChat estima cobertura a partir da ortofoto (GT raster), o MAE ainda é 17,4 pp. Se a ortofoto é o ground truth, por que o erro não é zero?

**Resposta dada:** Não soube responder.

**Resposta completa:**
Existem **dois ground truths distintos** no pipeline: o GT raster (ortofoto, melhor imagem de entrada) e o GT vetorial (GeoPackage, referência quantitativa para o MAE). O GeoChat erra 17,4 pp ao estimar percentuais a partir da ortofoto porque: (1) o modelo tem erro de estimativa intrínseco — é o **piso do sistema**, independente da resolução da imagem; (2) há ruído de anotação entre GeoPackage e ortofoto; (3) heterogeneidade dos tiles persiste mesmo a 1 m/px. Os 17,4 pp são a **calibração do sistema de medição**, não uma falha. Sem esse piso, não se pode interpretar os valores de LR e SR: a faixa recuperável é 22,7 → 17,4 = 5,3 pp; a SR recupera 3,6 pp, fechando ~68% da lacuna recuperável.

**Frase-chave:** *"O MAE de 17,4 pp da ortofoto é a calibração do sistema. Ele me diz o teto de precisão do GeoChat nessa tarefa — sem esse número, não sei se 3,6 pp é muito ou pouco."*

---

### Resumo das notas da simulação

| # | Tema | Nota | Situação |
|---|---|---|---|
| P1 | PSNR/SSIM vs PLN | 6/10 | Resposta parcial — faltou perception-distortion tradeoff |
| P2 | Heterogeneidade dos tiles | 5/10 | Faltou threshold e solução do VQA×GeoPackage |
| P3 | TF-IDF no Cenário A | 5/10 | Faltou transformar resultado fraco em argumento forte |
| P4 | Viés do RemoteCLIP | — | Não soube — ver Cenário C como controle |
| P5 | Água na Região 2 | 5/10 | Faltou mecanismo e implicação prática |
| P6 | VQA×GeoPackage como PLN | — | Não soube — preditor é linguagem natural |
| P7 | Confundimento SR vs fusão temporal | 3/10 | Deflectou limitação real — ponto crítico |
| P8 | Gap menor do GeoChat | — | Não soube — gap menor = resultado mais honesto |
| P9 | Objeção multi-região | — | Não soube — inversão da água prova diferença real |
| P10 | MAE da ortofoto ≠ 0 | — | Não soube — dois GT distintos, piso do modelo |

**Padrão de atenção para a banca:** limitações metodológicas (P7 especialmente), mecanismos concretos por trás dos resultados (P5), e a distinção entre os dois ground truths (P10). Esses temas exigem estudo adicional antes da defesa.

---

## 6. Explicação das siglas com exemplos

> Para cada termo: intuição central + exemplo concreto + papel no trabalho.

---

### BLEU — *Bilingual Evaluation Understudy*

**Ideia:** mede precisão de n-gramas — das palavras que o modelo gerou, quantas aparecem na referência?

**BLEU-1** usa apenas unigramas (palavras isoladas):

```
Gerado:     "a green forest with dense trees"
Referência: "forest"

Palavras do gerado que aparecem na referência: "forest" → 1
Total de palavras geradas: 6
BLEU-1 = 1/6 ≈ 0,17
```

Pune textos longos com palavras fora da referência. Orientado à **precisão**.

**No trabalho:** compara legendas BLIP/GeoChat com o GT (nome da classe ou legenda da ortofoto). No Cenário A, scores ≈ 0 porque a referência é só "forest" e o modelo descreve a cena com outras palavras.

---

### ROUGE-1 — *Recall-Oriented Understudy for Gisting Evaluation*

**Ideia:** mede revocação de unigramas — das palavras da referência, quantas o modelo cobriu? Reportado como **F1** (média harmônica de precisão e revocação).

```
Gerado:     "a green forest with dense trees"
Referência: "forest"

Palavras da referência cobertas pelo gerado: "forest" → 1
Total de palavras na referência: 1
Revocação = 1/1 = 1,0   |   Precisão = 1/6 ≈ 0,17
F1 = 2 × (1,0 × 0,17) / (1,0 + 0,17) ≈ 0,29
```

**Diferença em relação ao BLEU:** BLEU pergunta "o gerado é preciso?"; ROUGE pergunta "a referência foi coberta?".

**No trabalho:** mesma aplicação que o BLEU. Os dois são léxicos — exigem sobreposição exata de palavras, sem entender sinonímia. "forest" e "woodland" são palavras independentes para ambos.

---

### TF-IDF — *Term Frequency–Inverse Document Frequency*

**Ideia:** representa cada texto como um vetor onde cada dimensão é uma palavra, ponderada por:
- **TF** (frequência do termo no texto): palavras que aparecem mais têm peso maior.
- **IDF** (raridade no corpus): palavras comuns ("a", "the", "is") têm peso menor; palavras raras e discriminativas têm peso maior.

A similaridade entre dois textos é o **cosseno** entre seus vetores (valor entre 0 e 1).

```
Gerado:     "dense forest with tall trees"     → vetor [forest:0.8, trees:0.6, dense:0.4, ...]
Referência: "forest area with vegetation"      → vetor [forest:0.8, area:0.5, vegetation:0.6, ...]

Similaridade cosseno = produto dos vetores / (norma A × norma B) ≈ 0,45
```

**Limitação:** "forest" e "woodland" são dimensões completamente diferentes — não captura sinonímia.

**No trabalho:** métrica léxica mais robusta que BLEU/ROUGE porque usa IDF (reduz peso de palavras genéricas) e cosseno (normaliza tamanho do texto). É a principal métrica léxica nos resultados NLP.

---

### BLIP — *Bootstrapping Language-Image Pre-training*

**Ideia:** modelo de visão-linguagem pré-treinado em pares imagem-texto da Web. Dada uma imagem, gera uma legenda em linguagem natural.

```
Entrada:  [imagem de satélite com campos agrícolas]
Saída:    "a aerial view of farmland with green fields"
```

**Problema no domínio de SR:** treinado na Web (fotos comuns), quando a imagem LR tem baixa resolução e é ambígua, o BLIP recorre a cenas frequentes do seu treinamento e **alucina**:
```
Entrada LR ambígua → "a couple is kissing in the water"   ← alucinação
Entrada SR nítida  → "a green field with agricultural land"  ← correto
```

**No trabalho:** usado no Eixo 1 (NLP). Revela alucinações na LR (20% → 11% com SR), mas infla o gap justamente porque colapsa na LR.

---

### CLIP — *Contrastive Language-Image Pre-training*

**Ideia:** aprende, por *contrastive learning*, um espaço vetorial compartilhado entre imagem e texto. Pares (imagem, legenda) corretos ficam próximos; pares errados ficam distantes.

```
Encoder de imagem:  [foto de floresta] → vetor [0.2, 0.8, 0.1, ...]
Encoder de texto:   "a dense forest"   → vetor [0.3, 0.7, 0.2, ...]

Cosseno ≈ 0,92  →  alta similaridade
```

```
Encoder de imagem:  [foto de floresta] → vetor [0.2, 0.8, 0.1, ...]
Encoder de texto:   "an urban street"  → vetor [0.9, 0.1, 0.7, ...]

Cosseno ≈ 0,11  →  baixa similaridade
```

**No trabalho:** encoder visual do CLIP é usado de duas formas: (a) extraindo embeddings de imagem para comparar LR vs SR vs ortofoto diretamente (modo imagem-imagem); (b) como encoder do GeoChat.

---

### RemoteCLIP

**Ideia:** CLIP especializado para sensoriamento remoto. Foi fine-tuned nos datasets RSICD, RSITMD e UCM (pares imagem-texto de imagens de satélite). O espaço de embeddings é mais discriminativo para classes de cobertura do solo do que o CLIP genérico.

```
CLIP genérico:  [imagem de lavoura SR] vs [imagem de lavoura GT]  →  cosseno = 0,72
RemoteCLIP:     [imagem de lavoura SR] vs [imagem de lavoura GT]  →  cosseno = 0,91
```

Mais sensível porque "viu" imagens de satélite durante o treinamento.

**No trabalho:** métrica mais forte do Eixo 2 — SR > LR em **952/953 tiles**, gap de +0,508 (pooled). Resultado mais estável entre regiões de todo o pipeline.

---

### CLIPScore

**Ideia:** métrica de similaridade baseada em CLIP, sem precisar de referência textual humana. Fórmula:

```
CLIPScore = 2,5 × max(cosseno(embed_A, embed_B), 0)
```

- O fator **2,5** espalhasse os valores (similaridades CLIP ficam numa faixa estreita).
- O **max(..., 0)** descarta similaridades negativas (raras e sem sentido prático).

**Dois modos de uso no trabalho:**
- **Imagem-texto:** embedding da imagem vs embedding da legenda gerada.
- **Imagem-imagem:** embedding da LR/SR vs embedding da ortofoto — sem depender de modelo de legendagem. É o modo mais limpo e o que deu o resultado mais forte.

**No trabalho:** usado no Eixo 2. O CLIPScore imagem-imagem é a métrica que menos depende de ruído de geração de texto.

---

### SBERT / MiniLM — *Sentence-BERT* / *all-MiniLM-L6-v2*

**Ideia:** rede siamesa baseada em BERT que projeta sentenças inteiras num espaço vetorial onde o cosseno reflete **significado**, não sobreposição de palavras. Captura sinonímia e paráfrase.

```
TF-IDF:  "forest" vs "woodland"  →  similaridade = 0,0  (palavras diferentes)
SBERT:   "forest" vs "woodland"  →  similaridade ≈ 0,87 (significados próximos)
```

```
SBERT:   "There are many trees" vs "Dense forest coverage"  →  ≈ 0,79
TF-IDF:  "There are many trees" vs "Dense forest coverage"  →  ≈ 0,12
```

**No trabalho:** usado como "similaridade semântica" nas respostas VQA — complementa o TF-IDF. Quando o Δ semântico (SBERT) é menor que o Δ léxico (TF-IDF), indica que boa parte do ganho é reformulação de palavras, não mudança de conteúdo.

---

### GeoChat

**Ideia:** modelo visão-linguagem com arquitetura LLaVA (encoder visual CLIP + decodificador LLaMA-7B), **fine-tuned em 318 mil pares de instrução geoespaciais**. Responde perguntas sobre imagens de satélite em linguagem natural.

```
Entrada:  [imagem de satélite] + "Estimate the percentage of each land cover type"
Saída LR: "forest 30%, agriculture 40%, urban 20%, bare soil 10%"
Saída SR: "forest 35%, agriculture 45%, urban 15%, bare soil 5%"
```

Não alucina na LR como o BLIP porque foi treinado especificamente no domínio.

**No trabalho:** modelo central dos Eixos 2 e 3. VQA estratificada (7 perguntas, 3 níveis) e contribuição principal (VQA × GeoPackage).

---

### MAE — *Mean Absolute Error* (Erro Absoluto Médio)

**Ideia:** média dos erros absolutos entre valores estimados e valores reais.

```
GeoChat estima para um tile:  forest=30%, agriculture=50%, water=20%
GeoPackage (real):            forest=45%, agriculture=40%, water=5%

Erros absolutos: |30-45|=15, |50-40|=10, |20-5|=15
MAE = (15 + 10 + 15) / 3 = 13,3 pp
```

Unidade: **pontos percentuais (pp)** — diretamente interpretável ("erro médio de X pp na estimativa de cobertura").

**No trabalho:** métrica da contribuição central (VQA × GeoPackage). Compara estimativas do GeoChat com o GeoPackage por classe e globalmente. Resulta na hierarquia: GT raster 17,4 pp < SR 19,1 pp < LR 22,7 pp.

---

## 7. Explicação das tabelas da apresentação

---

### Slide 6 — CLIPScore imagem-imagem (Eixo 2)

```
Modelo       | Região | Δ      | SR > LR
-------------|--------|--------|--------
CLIP gen.    | R1     | +0,356 | 562/575
             | R2     | +0,345 | 368/378
             | pooled | +0,352 | 930/953
RemoteCLIP   | R1     | +0,505 | 574/575
             | R2     | +0,512 | 378/378
             | pooled | +0,508 | 952/953
```

**Colunas:**
- **Modelo:** qual encoder CLIP calcula a similaridade — CLIP genérico (treinado na Web) ou RemoteCLIP (fine-tuned em sensoriamento remoto).
- **Região:** R1 (575 tiles), R2 (378 tiles), pooled (953 tiles juntos).
- **Δ:** diferença média de CLIPScore: `CLIPScore(SR, ortofoto) − CLIPScore(LR, ortofoto)`. Positivo = SR ficou mais próxima da ortofoto.
- **SR > LR:** em quantos tiles individuais a SR ganhou da LR — contagem pareada tile a tile.

**Como ler:**
- RemoteCLIP R2 (378/378): a SR venceu em *todos* os tiles da segunda região.
- RemoteCLIP pooled (952/953): apenas 1 tile no universo inteiro favoreceu a LR.
- RemoteCLIP tem Δ maior que CLIP genérico porque seu espaço de embeddings é mais sensível a classes de cobertura do solo — foi treinado em imagens de satélite.

**Mensagem:** a SR é sistematicamente mais parecida com a ortofoto do que a LR, nas duas regiões, com os dois modelos. O RemoteCLIP quantifica esse ganho com mais precisão por ser do domínio.

---

### Slide 7 — VQA estratificada com GeoChat (Eixo 3)

**Tabela esquerda — as 7 perguntas em 3 níveis:**

```
Nível           | Pergunta
----------------|---------------------------------------------------
L1 Semântico    | "Estimate the percentage of each land cover type"
                | "Is there a water body visible in the image?"
L2 Estrutural   | "How many distinct agricultural fields can you count?"
                | "Are crop rows or cultivation patterns visible?"
L3 Detalhe fino | "How many individual buildings are visible?"
                | "Describe the road network visible in the image"
                | "Describe the buildings visible in the image"
```

Os níveis representam escala crescente de dificuldade para a resolução:
- **L1:** o que é o lugar? Qualquer resolução razoável responde.
- **L2:** como está organizado? Precisa de mais detalhe.
- **L3:** o que existe de específico? Só resolução alta responde bem.

**Tabela direita — resultados por nível:**

```
Nível           | TF-IDF Δ | Semântica Δ
----------------|----------|------------
L1 Semântico    | +0,228   | +0,054
L2 Estrutural   | +0,131   | +0,044
L3 Detalhe fino | +0,122   | +0,056
```

- **TF-IDF Δ:** ganho léxico das respostas SR vs LR frente ao GT.
- **Semântica Δ:** mesmo ganho medido por SBERT (significado, não palavras).

**Como ler linha por linha:**
- **L1:** maior ganho léxico — a SR faz o GeoChat usar vocabulário mais específico (nomes de classes, percentuais) que bate melhor com o GT. Ganho semântico menor: a LR já descrevia a cena de forma aproximadamente correta em significado.
- **L2:** ganho menor — pergunta q03 (contar campos) saturou em zero para ambas as resoluções, puxando o Δ do nível para baixo.
- **L3:** ganho semântico *maior* dos três níveis (+0,056). Perguntas sobre prédios e estradas dependem de detalhe fino — a SR muda de fato o *conteúdo* das respostas, não só o vocabulário.

**Por que TF-IDF Δ >> Semântica Δ em todos os níveis:** parte do ganho léxico é reformulação de vocabulário (a SR induz o GeoChat a usar termos mais técnicos que casam com o GT). O SBERT captura o ganho real de compreensão, que é mais modesto e mais honesto. Reportar os dois lado a lado torna o pipeline autocrítico.

**Mensagem:** a SR melhora as respostas em todos os níveis; o ganho semântico real é mais forte em L3 (detalhe fino), exatamente onde a resolução mais importa.

---

### Slide 9 — Resultado principal: MAE de cobertura LR vs SR vs GT

```
Classe      | R1 LR | R1 SR | R2 LR | R2 SR | Pool LR | Pool SR
------------|-------|-------|-------|-------|---------|--------
forest      |  37,1 |  36,6 |  28,8 |  23,9 |    33,8 |    31,6
agriculture |  30,4 |  26,1 |  50,1 |  41,0 |    38,3 |    31,9
urban       |   5,1 |   3,7 |   5,0 |   5,8 |     5,1 |     4,5
water       |  11,5 |   9,6 |   4,7 |  13,4 |     8,8 |    11,1
bare soil   |  29,4 |  20,2 |  24,4 |  10,3 |    27,4 |    16,4
------------|-------|-------|-------|-------|---------|--------
Global      |  22,7 |  19,2 |  22,6 |  18,9 |    22,7 |    19,1
GT raster   |    17,7     |     16,9      |       17,4
```

**Unidade: pontos percentuais (pp). Menor = melhor.**

**Estrutura:** cada par de colunas (LR / SR) mostra o MAE do GeoChat ao estimar cobertura do solo a partir de cada resolução. A diferença LR − SR é o ganho da SR. GT raster é o teto alcançável (erro intrínseco do modelo com a melhor imagem possível).

**Classe por classe:**

- **bare soil — maior ganho:** R1 −9,2 pp / R2 −14,1 pp. Solo exposto tem textura e contraste que a fusão temporal recupera bem. Robusto nas duas regiões.
- **agriculture:** R1 −4,3 pp / R2 −9,1 pp. R2 começa mais alto (paisagem mais agrícola) e o ganho é maior.
- **forest:** ganho modesto em R1 (−0,5 pp), mais expressivo em R2 (−4,9 pp). Floresta é textura homogênea — mais fácil de estimar mesmo na LR.
- **urban:** valores baixos porque área urbana é minoria. Piora de +0,8 pp em R2 é estatisticamente insignificante dado o pequeno número de casos.
- **water — achado de falha:** R1 melhora (−1,9 pp); R2 piora muito (+8,7 pp). Em R2 a água é rara — a SR alucina água onde não há, e a LR acertava estimando ~0%. É a inversão mais dramática da tabela.

**Linha Global e GT raster:**
```
LR 22,7 → SR 19,1 → GT raster 17,4  (pooled)
```
- Faixa recuperável: 22,7 − 17,4 = 5,3 pp
- SR recupera 3,6 pp → fecha **~68% da lacuna recuperável**
- GT raster não é zero porque o modelo tem erro intrínseco (dois GT distintos: ortofoto como imagem de entrada, GeoPackage como referência quantitativa)

**Mensagem:** a SR melhora a estimativa de cobertura em quase todas as classes e regiões; o único modo de falha sistemático é a classe water em paisagens onde a água é rara.

---

### Slide 12 — Resumo: o que cada eixo mede

```
Eixo                 | O que mede                     | Principal achado
---------------------|--------------------------------|------------------------------------------
BLIP NLP             | Qualidade léxica das legendas  | SR reduz alucinações de 20% → 11%
CLIPScore RemoteCLIP | Similaridade semântica img-img | SR > LR em 952/953 tiles
GeoChat NLP          | Qualidade com modelo robusto   | Gap menor confirma que BLIP infla resultado
VQA estratificada    | Informação por nível           | Todos os níveis favorecem SR
VQA × GeoPackage     | Erro de cobertura em pp        | −3,6 pp global; maior em bare soil e agriculture
```

**Linha por linha:**

- **BLIP NLP:** útil como detector de colapso na LR. O achado central não é o score — é que 20% dos tiles LR alucinam, e a SR reduz para 11%.
- **CLIPScore RemoteCLIP:** eixo mais limpo metodologicamente (sem legendagem, sem texto). 952/953 tiles favoráveis é o resultado mais expressivo e fácil de comunicar.
- **GeoChat NLP:** gap menor não é fraqueza — é validação cruzada. Confirma que parte do gap do BLIP era artefato de alucinação na LR.
- **VQA estratificada:** perguntas objetivas em 3 níveis de dificuldade. Todos os níveis favorecem SR — o ganho não está concentrado em apenas um tipo de informação.
- **VQA × GeoPackage** *(contribuição central, em negrito no slide):* única métrica ao mesmo tempo semântica e quantitativamente interpretável. −3,6 pp global é o número de manchete do trabalho.

**Por que a ordem importa — progressão lógica do pipeline:**
```
BLIP NLP         → revela alucinações, mas infla gap
       ↓
CLIPScore        → confirma SR > LR sem depender de texto
       ↓
GeoChat NLP      → gap menor, resultado mais honesto
       ↓
VQA estratificada → discrimina por nível, mais controlado
       ↓
VQA × GeoPackage → elimina dependência léxica, dá pp direto
```
Cada eixo corrige uma limitação do anterior. A tabela do slide 12 é o mapa de como o trabalho se constrói.
