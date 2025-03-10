from transformers import BertTokenizer

# BERT 기본 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Treatise 텍스트
text = """
Innitagnostus ÖPIK, 1967a, p. 98 [*I. innitens; OD;
holotype (ÖPIK, 1967, pl. 58, fig. 2), CPC 5853,
AGSO, Canberra]. Median preglabellar furrow variably
developed. Glabella with broad, trapeziform
anterior lobe; F3 well impressed, nearly straight;
posterior lobe with well-developed F1 and F2; lateral
portions of M2 commonly separated from
midmost glabella by weak longitudinal (exsag.) furrows;
glabellar node located from midway between
F1 and F2 to level with F2; basal lobes of moderate
size, trapezoidal. Pygidial axis of moderate length,
constricted across M2; M1 trilobate; F1 well impressed,
bent forward; F2 straight laterally, bent
rearward by strong axial node. Posterior lobe ogival
to semiovate, usually narrowly rounded posteriorly,
not reaching border furrow. Median postaxial furrow
absent. Upper Cambrian: China (Guizhou,
Hunan); Australia (Queensland), Mindyallan–Idamean
(E. eretes to S. diloma Zones); Russia (Siberia),
Kazakhstan, G. stolidotus to P. curtare Zones;
Canada (Northwest Territories, British Columbia,
Newfoundland), Glyptagnostus reticulatus to
Olenaspella regularis Zones; USA (Alabama, Nevada,
Texas), Aphelaspis Zone.——FIG. 217,6a,b.
*I. innitens, Upper Cambrian (Mindyallan, G.
stolidotus Zone), western Queensland (Boulia district);
a, holotype, cephalon, CPC 5853, ×8; b,
topotype, pygidium, CPC 5854, ×8 (new).
"""

# 토큰화 수행
tokens = tokenizer.tokenize(text)

# 토큰 개수 출력
print(f"Token 개수: {len(tokens)}")
