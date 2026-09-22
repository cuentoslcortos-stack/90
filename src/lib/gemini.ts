import { GoogleGenAI, ThinkingLevel } from "@google/genai";

export const GEMINI_MODEL = "gemini-3.6-flash";

/**
 * Identifica a la materia y sirve como anclaje en la UI
 * (panel de Configuración muestra este string).
 */
export const ASSISTANT_LABEL =
  "Asistente 90A — Biología del Comportamiento (Cód. 90A, Cátedra Muzio — Dr. Rubén N. Muzio, UBA Psicología)";

/**
 * Base de conocimiento de la materia: PDFs servidos como archivos
 * estáticos en `public/` y subidos a Gemini File API en runtime.
 *
 * El modelo NO debe responder con nada que no esté en estos dos PDFs.
 * Se suben a Gemini File API una sola vez por sesión y se referencian
 * por fileUri (cache 24 h en localStorage).
 */
const VITE_BASE_URL: string =
  ((import.meta as ImportMeta & { env: Record<string, string | undefined> }).env?.BASE_URL ?? "/");

const PDF_SOURCES: ReadonlyArray<{ name: string; path: string }> = [
  { name: "01.BC_1P.pdf", path: `${VITE_BASE_URL}01.BC_1P.pdf` },
  { name: "02.BC_2P.pdf", path: `${VITE_BASE_URL}02.BC_2P.pdf` },
] as const;

/**
 * Cache en localStorage: para cada PDF guardamos { uri, expiry }.
 * TTL: 24h (la File API de Gemini expira a las 48h, dejamos margen).
 */
const KB_CACHE_PREFIX = "gem-pdf-uri:";
const KB_TTL_MS = 24 * 60 * 60 * 1000;

interface CachedUri {
  uri: string;
  expiry: number;
}

function readCachedUri(name: string): string | null {
  try {
    const raw = localStorage.getItem(KB_CACHE_PREFIX + name);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CachedUri;
    if (!parsed?.uri || !parsed?.expiry) return null;
    if (parsed.expiry < Date.now()) return null;
    return parsed.uri;
  } catch {
    return null;
  }
}

function writeCachedUri(name: string, uri: string): void {
  try {
    const payload: CachedUri = { uri, expiry: Date.now() + KB_TTL_MS };
    localStorage.setItem(KB_CACHE_PREFIX + name, JSON.stringify(payload));
  } catch {
    /* sin persistencia: se re-subirá cada vez */
  }
}

/**
 * Sube un PDF a Gemini File API y devuelve el fileUri.
 * Si ya hay uno cacheado en localStorage (no expirado), lo reusa.
 */
async function ensurePdfUploaded(
  ai: GoogleGenAI,
  name: string,
  path: string
): Promise<string> {
  const cached = readCachedUri(name);
  if (cached) return cached;

  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`No se pudo cargar ${name} desde el sitio (HTTP ${res.status}).`);
  }
  const blob = await res.blob();
  if (blob.size === 0) {
    throw new Error(`El archivo ${name} está vacío.`);
  }

  const uploaded = await ai.files.upload({
    file: new File([blob], name, { type: "application/pdf" }),
    config: { displayName: name },
  });
  const uri = uploaded?.uri;
  if (!uri) {
    throw new Error(`No se pudo subir ${name} a Gemini File API.`);
  }
  writeCachedUri(name, uri);
  return uri;
}

/**
 * Prepara la base de conocimiento: sube los PDFs declarados en
 * `PDF_SOURCES` a Gemini File API (o reutiliza los URIs cacheados)
 * y devuelve un array de `fileData` listo para meter en `parts[]`.
 *
 * Si `PDF_SOURCES` está vacío, lanza un error claro para que el
 * usuario sepa que falta cargar el material antes de usar la app.
 */
async function buildKnowledgeBaseParts(
  ai: GoogleGenAI,
  onProgress?: (msg: string) => void
): Promise<{ fileData: { fileUri: string; mimeType: string } }[]> {
  if (PDF_SOURCES.length === 0) {
    throw new Error(
      "La base de conocimiento está vacía. Copiá los PDFs a public/ y " +
        "registralos en PDF_SOURCES dentro de src/lib/gemini.ts."
    );
  }
  const parts: { fileData: { fileUri: string; mimeType: string } }[] = [];
  for (const src of PDF_SOURCES) {
    onProgress?.(`Subiendo ${src.name} a Gemini…`);
    const uri = await ensurePdfUploaded(ai, src.name, src.path);
    parts.push({ fileData: { fileUri: uri, mimeType: "application/pdf" } });
  }
  return parts;
}

/**
 * Prompt del sistema — Tutor 90A, materia "Biología del Comportamiento"
 * (Código 90A, Cátedra Dr. Rubén N. Muzio, Facultad de Psicología, UBA).
 */
export const SYSTEM_PROMPT = `# SYSTEM PROMPT: TUTOR IA DE BIOLOGÍA DEL COMPORTAMIENTO (CÁTEDRA MUZIO)
Rol: Tutor IA experto en Biología del Comportamiento (Cátedra Dr. Rubén N. Muzio, Psicología UBA).
Objetivo: Resolver preguntas Multiple Choice, y de "Verdadero o Falso", utilizando ÚNICA Y EXCLUSIVAMENTE los documentos provistos: "01.BC_1P.pdf" (Primer parcial) y "02.BC_2P.pdf" (Segundo parcial).
CONTEXTO DE EVALUACIÓN Y TEMARIO - Exámenes exigentes de 30 preguntas Multiple Choice (hasta 5 opciones, incluyendo "Todas/Ninguna") o Verdadero/Falso con trampas conceptuales.
---
### 1. REGLA FUNDAMENTAL DE FUENTE CERRADA (STRICT GROUNDING)
* Trabajarás exclusivamente sobre la información contenida en:
  - 01.BC_1P.pdf: Para todos los contenidos correspondientes al 1° Parcial.
  - 02.BC_2P.pdf: Para todos los contenidos correspondientes al 2° Parcial.
* No inventarás datos, ni extrapolarás conocimientos generales de biología o psicología externa que contradigan o no figuren en los textos y fichas de la cátedra.
* Si una afirmación no está sustentada por el material provisto, se considerará inválida o no respaldada en el marco de la cátedra.
---
### 2. MAPEO CURRICULAR Y DIVISIÓN DE CONTENIDOS
Debes identificar con precisión a qué bloque pertenece la consulta para remitirte al documento correspondiente:
#### BLOQUE 1: PRIMER PARCIAL (01.BC_1P.pdf)
* Metodología e Introducción al Comportamiento:
  - Concepto de Biología del Comportamiento.
  - Enfoque monista psicobiológico frente al dualismo.
  - Los cuatro niveles de causalidad (Tinbergen/Mayr): Causas próximas (fisiología/mecanismos y ontogenia/desarrollo) y Causas últimas (función adaptativa y filogenia/evolución).
  - Método científico, diseño experimental (VI, VD, variables controladas, interacciones) y técnicas etológicas de registro.
  - Autores/Fichas: Freidin & Muzio; Papini (Cap. 1).
* Análisis Comparado y Evolución del Sistema Nervioso:
  - Evolución del sistema nervioso en vertebrados; encéfalo, corteza y palio medial.
  - Técnicas neurobiológicas: ablaciones/lesiones, estimulación cerebral, registros electrofisiológicos y neuroimágenes modernas.
  - Autores/Fichas: Muzio (adaptación Rosenzweig Cap. 3).
* Estrés:
  - Definición psicofisiológica del estrés (Selye, McEwen). Síndrome General de Adaptación.
  - Mecanismos neuroinmunoendocrinos: Activación del Sistema Nervioso Autónomo Simpático (médula adrenal - catecolaminas: adrenalina y noradrenalina) vs. Eje HPA (hipotálamo-hipófiso-adrenal: CRH, ACTH, glucocorticoides/cortisol/corticosterona).
  - Variables psicológicas moduladoras: Controlabilidad y Predictibilidad. Indefensión aprendida.
  - Indicadores de estrés (fisiológicos, bioquímicos, conductuales).
  - Autores/Fichas: Daneri; Pompilio; Muzio (adaptación Kalat Cap. 12).
* Ansiedad:
  - Bases psicobiológicas de la ansiedad. Neuroanatomía del miedo y la ansiedad (amígdala, hipocampo, corteza prefrontal).
  - Sistemas de neurotransmisión (GABA, serotonina, noradrenalina).
  - Autores/Fichas: Muzio (adaptación Kalat Cap. 12).
* Depresión:
  - Modelos psicobiológicos de la depresión mayor. Hipótesis monoaminérgica (serotonina, noradrenalina, dopamina).
  - Mecanismos de acción de psicofármacos antidepresivos (IMAO, tricíclicos, ISRS).
  - Modelos animales de depresión e indefensión aprendida (Seligman).
  - Autores/Fichas: Mandich, Grinspun & Muzio.
* Bases Neurobiológicas de Trastornos Mentales / Patologías:
  - Esquizofrenia: Alteraciones estructurales (ventriculomegalia, hipofrontalidad) y funcionales; hipótesis dopaminérgica y del neurodesarrollo. Mecanismo de neurolépticos/antipsicóticos.
  - Demencia senil / Alzheimer: Procesos neurodegenerativos, placas amiloides, ovillos neurofibrilares, sistema colinérgico.
  - Autores/Fichas: Daneri & Muzio; Muzio (adaptación Rosenzweig Cap. 15).
* Motivación: Anorexia y Bulimia:
  - Modelos motivacionales, impulsos (drives), saciedad y hambre.
  - Regulación neuroendocrina: Hipotálamo lateral (hambre) e Hipotálamo ventromedial (saciedad); leptina, grelina, neuropéptido Y.
  - Trastornos de la conducta alimentaria y modelos de experimentación.
  - Autores/Fichas: Daneri; Bridgeman (Cap. 9).
* Toma de Decisiones y Heurísticos:
  - Racionalidad acotada (Simon, Kahneman, Tversky).
  - Heurísticos de juicio: Representatividad, Disponibilidad, Anclaje y Ajuste. Sesgos cognitivos.
  - Preferencias contexto-dependientes y estado-dependientes. Modelos comparados en animales y humanos.
  - Autores/Fichas: Squillace; Pompilio.
---
#### BLOQUE 2: SEGUNDO PARCIAL (02.BC_2P.pdf)
* Selección Sexual y Origen de los Sexos:
  - Definición evolutiva de los sexos: Anisogamia y sus consecuencias biológicas.
  - Teoría de la Inversión Parental (Trivers).
  - Selección intrasexual (competencia entre miembros del mismo sexo, armamentos) vs. Selección intersexual (elección de pareja, ornamentos, hipótesis de buenos genes, principio del hándicap de Zahavi).
  - Estrategias reproductivas humanas y patrones de emparejamiento.
  - Autores/Fichas: Gabelli; TP Selección Sexual.
* Genética del Comportamiento y Modelo de los Caminos:
  - Influencia genética y ambiental sobre el comportamiento.
  - Heredabilidad (h^2): Concepto poblacional, cuantitativo y de varianza (no aplicable a nivel individual; no determina inmutabilidad).
  - Diseños metodológicos: Estudios de gemelos (monocigóticos vs. dicigóticos) y estudios de adopción.
  - Modelo de los caminos (Path Analysis): Estimación de componentes genéticos (h^2), ambiente compartido (c^2) y ambiente no compartido (e^2). Correlación e interacción gen-ambiente.
  - Genética de rasgos complejos (personalidad, CI) y patologías.
  - Autores/Fichas: Gabelli; Simonetti; Plomin et al.
* Desarrollo del Comportamiento (Ontogenia):
  - Superación de la falsa dicotomía Instinto-Aprendizaje.
  - Modelo epigenético: Interacción dinámica bidireccional gen-ambiente durante el desarrollo.
  - Agentes del cambio ontogénico: Genes, maduración, experiencia y aprendizaje.
  - Períodos sensibles/críticos e impronta (Lorenz, Bateson).
  - Crítica a la ley biogenética de Haeckel (recapitulación) y programa somático (Mayr, Gould).
  - Autores/Fichas: Gabelli.
* Origen y Evolución del Lenguaje:
  - Características del lenguaje humano: Sintaxis, recursión, doble articulación, simbolismo vs. sistemas de comunicación animal.
  - El lenguaje como adaptación biológica producto de la selección natural (Pinker & Bloom) vs. Spandrel/subproducto (Gould & Lewontin).
  - Modelos con primates no humanos (Washoe, Kanzi, Koko): alcances y limitaciones semánticas/sintácticas.
  - Protolenguajes (Bickerton). Bases genéticas (gen FOXP2) y patologías del lenguaje.
  - Autores/Fichas: Maynard Smith & Szathmary (Cap. 17); Pinker; Töpf & Simonetti.
---
### 3. PROTOCOLO DE RESOLUCIÓN DE EXÁMENES (TRAMPAS Y DISTRACTORES)
La Cátedra Muzio formula preguntas con alto nivel de discriminación conceptual. Para cada ítem debes estar alerta a las trampas recurrentes:
1. Confusión de Causas de Tinbergen: Cuidarse de opciones que explican un "para qué" (causa última/función adaptativa) cuando la pregunta pide el "cómo" o mecanismo fisiológico inmediato (causa próxima), o viceversa.
2. Trampas sobre Heredabilidad (h^2):
   - Falso: "Una heredabilidad de 0.80 significa que el 80% del rasgo en un individuo se debe a los genes".
   - Correcto: Es una medida estadística poblacional referida a la varianza fenotípica explicada por la varianza genética en una población y ambiente específicos.
3. Determinismo Genético vs. Epigénesis: Rechazar opciones preformistas o radicalmente ambientalistas. El desarrollo es siempre interactivo (modelo epigenético).
4. Distinción Fisiológica del Estrés:
   - La respuesta rápida/inmediata es simpático-médulo-adrenal (adrenalina/noradrenalina).
   - La respuesta lenta o sostenida corresponde a la corteza adrenal (cortisol/corticosterona vía ACTH). No confundir médula adrenal con corteza adrenal.
5. Opciones "Todas las anteriores son correctas" / "Ninguna es correcta":
   - No asumir automáticamente que son la respuesta correcta. Revisa una por una las premisas; si al menos una premisa es indudablemente falsa, descarta "Todas". Si al menos una es indudablemente verdadera, descarta "Ninguna".
---
### 4. ESTRUCTURA OBLIGATORIA DE RESPUESTA
Al recibir una pregunta (Multiple Choice o V/F), responderás estructurando el contenido exactamente de la siguiente manera: Toda respuesta debe respetar ESTRICTAMENTE la siguiente estructura fija, sin agregar introducciones, saludos, metadatos técnicos ni listas con viñetas en la justificación.
* **Extensión total obligatoria:** Entre 200 y 250 palabras.
* **Estructura en dos puntos:**
RESPUESTA:
1. Opción correcta: [Número o Letra]. [Texto o enunciado de la opción seleccionada].
2. Por qué las otras son incorrectas: [Párrafo único, fluido, denso y articulado conceptualmente. Debe explicar en prosa continua por qué se descartan las demás alternativas, identificando las trampas teóricas o distorsiones de los conceptos y contrastándolas directamente con los postulados oficiales de los autores de la cátedra].
*(En preguntas de Verdadero o Falso, adaptar el punto 1 a "1. Dictamen: [Verdadero / Falso]" y el punto 2 a "2. Justificación y análisis del error:", manteniendo la misma extensión y prosa unificada).*`;

/**
 * Nota legible sobre qué hay cargado como base de conocimiento.
 * Sólo se usa en logs / debug; el modelo la ignora.
 */
export const KNOWLEDGE_BASE_NOTE =
  "Base de conocimiento: 01.BC_1P.pdf (1P — Metodología, Evolución SN, Estrés, Ansiedad, Depresión, Patologías, Motivación, Decisiones) + 02.BC_2P.pdf (2P — Selección Sexual, Genética del Comportamiento, Ontogenia, Lenguaje) — subidos a Gemini File API.";

/**
 * Lee la API key desde la variable de entorno de Vite.
 * Se mantiene como fallback; la app prefiere siempre la key que el usuario
 * haya guardado en el panel de Configuración (localStorage).
 */
export const GEMINI_API_KEY: string = (
  (import.meta as ImportMeta & { env: Record<string, string | undefined> }).env?.VITE_GEMINI_API_KEY ?? ""
).trim();


export function pickMimeType(): string {
  if (typeof MediaRecorder === "undefined") return "audio/webm";
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
  ];
  return candidates.find((m) => MediaRecorder.isTypeSupported(m)) ?? "audio/webm";
}

/**
 * Convierte un Blob (audio grabado) a una cadena Base64 *sin* el prefijo
 * `data:<mime>;base64,` que agrega FileReader — es lo que espera Gemini
 * en `inlineData.data`.
 */
export function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result ?? "");
      resolve(result.split(",")[1] ?? "");
    };
    reader.onerror = () => reject(new Error("No se pudo codificar el audio a Base64."));
    reader.readAsDataURL(blob);
  });
}

/**
 * Limpia el texto que devuelve Gemini antes de mostrarlo o leerlo en voz
 * alta. Caza los artefactos típicos de cuando el modelo se "contagia" del
 * formato de transcripción de audio (timecodes SRT/VTT, etiquetas de
 * hablante, etc.) y de cualquier residuo de markdown que el TTS leería
 * literal (asteriscos, guiones bajos, etc.). Pensada como red de seguridad:
 * aunque el system prompt lo prohíba, el modelo a veces los emite igual.
 *
 * Patrones que elimina:
 *  - Sello MM:SS o HH:MM:SS pegado o suelto:           00:05 · 1:23 · 00:05.123
 *  - Pegado a una palabra (sin espacio):                "socio01:03estructural" → "socioestructural"
 *  - Con corchetes / ángulos / paréntesis:              [00:05] · <00:05> · (00:05)
 *  - Rangos SRT/VTT:                                    00:05 --> 00:08 · 00:05,000 --> 00:08,000
 *  - Etiquetas de hablante:                             Speaker 1: · Hablante 2:
 *  - Líneas que son solo un número (índices SRT)
 *  - Marcado Markdown simple: **negrita**, *itálica*, _itálica_, `código`
 */
export function sanitizeResponseText(text: string): string {
  if (!text) return text;
  let t = text;
  // 1) Índices de bloque SRT: una línea entera que es solo 1-4 dígitos
  t = t.replace(/^\s*\d{1,4}\s*$/gm, "");
  // 2) Rangos SRT/VTT: "00:05 --> 00:08" / "00:05,000 --> 00:08,000"
  t = t.replace(
    /\b\d{1,2}:\d{2}(?:[.,]\d{1,3})?\s*-->\s*\d{1,2}:\d{2}(?:[.,]\d{1,3})?\b/g,
    " "
  );
  // 3) Sellos de tiempo con corchetes/ángulos/paréntesis: [00:05], <1:23>
  t = t.replace(
    /[\[\<\(]\s*\b\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?\b\s*[\]\>\)]/g,
    " "
  );
  // 4) Sellos sueltos: 00:05, 1:23, 00:05.123 (incluye HH:MM:SS).
  //    Importante: NO usar \b al final, porque un sello pegado a una
  //    palabra ("socio01:03estructural") no tiene word boundary y el
  //    \b lo dejaría pasar. Usamos (?<!\d) al inicio (para no
  //    comernos el "12" de "12:00:30") y (?!\d) al final (para no
  //    comernos el "00" de "12:00:30.5"). El reemplazo es "" (sin
  //    espacio) para que el texto fluya al pegarse a la palabra.
  t = t.replace(/(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?:[.,]\d{1,3})?(?!\d)/g, "");
  // 5) Etiquetas de hablante: "Speaker 1:", "Hablante 2]", "Speaker1 -"
  t = t.replace(/\b(?:Speaker|Hablante|Unknown)\s*\d+\s*[:\-\]]\s*/gi, " ");
  // 6) Markdown residual: negrita (**), itálica (*) y código (`).
  //    El system prompt prohíbe markdown, pero a veces el modelo se
  //    "contagia" y lo emite igual — y speechSynthesis lo lee literal
  //    ("asterisco asterisco negrita asterisco asterisco").
  t = t.replace(/\*\*([^*]+)\*\*/g, "$1");
  t = t.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1$2");
  t = t.replace(/`([^`]+)`/g, "$1");
  // 6.5) Guiones largos / rayas (—, –) y secuencias de guiones
  //      enfáticos. El system prompt los prohíbe, pero el modelo
  //      a veces los emite como pausas dramáticas. speechSynthesis
  //      los lee literal ("guión guión guión..."). Los borramos como
  //      red de seguridad antes de la limpieza final.
  t = t.replace(/[—–]+/g, " ");
  // 7) Limpieza: colapsa espacios y saltos de línea sobrantes
  t = t.replace(/[ \t]{2,}/g, " ");
  t = t.replace(/[ \t]+\n/g, "\n");
  t = t.replace(/\n{3,}/g, "\n\n");
  return t.trim();
}

/** Extrae un mensaje legible de un error arbitrario (incluido el del SDK). */
function describeError(err: unknown): string {
  if (typeof err === "string") return err;
  if (err && typeof err === "object") {
    const e = err as {
      message?: string;
      status?: number | string;
      code?: number | string;
      error?: { message?: string; code?: number | string; status?: string };
    };
    if (e.error?.message) {
      const code = e.error.code ?? e.error.status ?? e.status ?? e.code;
      return code ? `[${code}] ${e.error.message}` : e.error.message;
    }
    if (e.message) return e.message;
  }
  return "Error desconocido al hablar con Gemini.";
}

/**
 * Detecta errores transitorios del servicio (503 UNAVAILABLE,
 * "high demand", "overloaded", etc.). En esos casos, reintentamos
 * una vez antes de mostrar el error al usuario.
 */
function isTransientError(err: unknown): boolean {
  const detail = describeError(err).toLowerCase();
  return (
    detail.includes("503") ||
    detail.includes("unavailable") ||
    detail.includes("high demand") ||
    detail.includes("overloaded") ||
    detail.includes("try again later")
  );
}

/**
 * Sube los PDFs de la bibliografía a Gemini File API (o reutiliza
 * los URIs cacheados en localStorage). Es idempotente: si el cache
 * expiró o nunca existió, sube; si todavía es válido, no hace nada.
 *
 * Útil para "calentar" la base de conocimiento al inicio de la sesión
 * y para que la UI pueda mostrar el estado ("Subiendo PDFs a Gemini…").
 */
export async function warmupKnowledgeBase(
  apiKey: string,
  onProgress?: (msg: string) => void
): Promise<void> {
  const cleanKey = apiKey.trim();
  if (!cleanKey || cleanKey === "TU_API_KEY_AQUI") {
    throw new Error("Configura tu API Key de Gemini en el panel de Configuración.");
  }
  const ai = new GoogleGenAI({ apiKey: cleanKey });
  await buildKnowledgeBaseParts(ai, onProgress);
}

/**
 * Indica si la base de conocimiento ya está cacheada y vigente.
 * Devuelve true si AMBOS PDFs tienen un fileUri no expirado.
 */
export function isKnowledgeBaseReady(): boolean {
  return PDF_SOURCES.every((s) => readCachedUri(s.name) !== null);
}

/**
 * Envía el audio + la base de conocimiento (PDFs vía File API) a Gemini
 * usando el SDK oficial `@google/genai`.
 *
 * Estructura del request:
 *   parts: [
 *     ...pdfFileData[],              // todos los PDFs en PDF_SOURCES
 *     { inlineData: <audio> },       // clip grabado
 *     { text: <instrucción> }        // "Escuchá el audio y respondé…"
 *   ]
 *
 * Manejo de errores:
 *  - Errores transitorios (503/UNAVAILABLE/"high demand"): reintenta una
 *    vez con 4 s de espera. Si el segundo intento también falla, muestra
 *    un mensaje claro en español.
 *  - API key inválida / 401/403: mensaje específico, sin reintento.
 *  - Cuota agotada / 429: mensaje específico, sin reintento.
 *  - Errores de red: mensaje específico, sin reintento.
 */
export async function askGemini(
  base64Audio: string,
  mimeType: string,
  apiKey: string,
  onProgress?: (msg: string) => void
): Promise<string> {
  const cleanKey = apiKey.trim();
  if (!cleanKey || cleanKey === "TU_API_KEY_AQUI") {
    throw new Error("Configura tu API Key de Gemini en el panel de Configuración.");
  }

  const ai = new GoogleGenAI({ apiKey: cleanKey });

  // 1) Base de conocimiento: sube los PDFs a File API (o reusa cache).
  onProgress?.("Preparando base de conocimiento…");
  const pdfParts = await buildKnowledgeBaseParts(ai, onProgress);

  const contents = [
    {
      parts: [
        ...pdfParts,
        { inlineData: { mimeType, data: base64Audio } },
        {
          text:
            "Escuchá el audio adjunto y respondé según las instrucciones del sistema. " +
            "Tu respuesta debe fundamentarse exclusivamente en los PDFs cargados " +
            "como base de conocimiento (ver PDF_SOURCES en src/lib/gemini.ts). " +
            "Ajustate al formato y la extensión definidos en el system prompt.",
        },
      ],
    },
  ];
  const config = {
    systemInstruction: SYSTEM_PROMPT,
    // 4096 tokens: en Gemini 3, los tokens de thinking cuentan contra
    // maxOutputTokens. Con este margen, el modelo tiene aire para
    // pensar (poco) y responder las 200-250 palabras que exige el
    // system prompt sin cortarse.
    maxOutputTokens: 4096,
    // Thinking MINIMAL = mínimo gasto de tokens en razonamiento
    // previo, deja el grueso del budget para la respuesta visible.
    // Con LOW se comía ~2300 tokens y dejaba la respuesta en ~100.
    thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
    // Temperatura baja = respuestas más deterministas y ligeramente
    // más rápidas (menos sampling).
    temperature: 0.3,
  };

  const MAX_ATTEMPTS = 2;
  let lastErr: unknown = null;
  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    try {
      const response = await ai.models.generateContent({
        model: GEMINI_MODEL,
        contents,
        config,
      });
      const text = (response?.text ?? "").trim();
      if (!text) {
        throw new Error("Gemini no devolvió texto. Intenta grabar la pregunta con más claridad.");
      }
      return text;
    } catch (err) {
      lastErr = err;
      if (attempt < MAX_ATTEMPTS && isTransientError(err)) {
        // Espera 4 s antes del reintento.
        await new Promise((resolve) => setTimeout(resolve, 4000));
        continue;
      }
      break;
    }
  }

  // Si llegamos acá, falló definitivamente. Mapeo a un mensaje en
  // español claro, sin JSON crudo en la UI.
  const detail = describeError(lastErr);
  const lower = detail.toLowerCase();
  if (
    lower.includes("api key") ||
    lower.includes("auth") ||
    lower.includes("credential") ||
    lower.includes("permission") ||
    lower.includes("401") ||
    lower.includes("403")
  ) {
    throw new Error(`API Key rechazada por Gemini: ${detail}`);
  }
  if (lower.includes("quota") || lower.includes("429") || lower.includes("rate")) {
    throw new Error(`Cuota o rate-limit de Gemini: ${detail}`);
  }
  if (isTransientError(lastErr)) {
    throw new Error(
      "El servicio de Gemini está saturado. Reintentá en unos minutos. " +
        `Detalle: ${detail}`
    );
  }
  if (lower.includes("network") || lower.includes("fetch") || lower.includes("econn") || lower.includes("timeout")) {
    throw new Error(`Sin conexión con Gemini: ${detail}`);
  }
  throw new Error(`Gemini rechazó la solicitud: ${detail}`);
}

/**
 * Transcribe LITERALMENTE el audio a texto (español rioplatense).
 *
 * Se usa SOLO para el log automático de Q&A (qa-logs/): corre en segundo
 * plano DESPUÉS de que la respuesta académica ya se mostró y leyó, así no
 * suma latencia a la UX. Llamada liviana: sin PDFs de la base de
 * conocimiento, pocos tokens, temperatura 0.
 *
 * Devuelve la transcripción verbatim (sin timecodes ni etiquetas de
 * hablante). Lanza si Gemini no devuelve texto — el llamador debe hacer
 * fallback a guardar el log sin transcripción, nunca mostrar error al alumno.
 */
/**
 * Prompt de transcripción: reforzado para impedir razonamiento verbal.
 * Lo que va entre `<<T>>...<</T>>` es lo único que la app va a leer;
 * si el modelo "piensa en voz alta", esa parte queda afuera.
 */
const TRANSCRIBE_INSTRUCTION =
  "Tu ÚNICA tarea es transcribir LITERALMENTE el audio adjunto al texto, en español.\n" +
  "REGLA ABSOLUTA: tu respuesta completa debe consistir EXCLUSIVAMENTE en la transcripción, " +
  "encerrada entre los marcadores `<<T>>` y `<</T>>`. No escribas nada fuera de esos marcadores.\n" +
  "PROHIBIDO terminantemente incluir: razonamientos, justificaciones, verificaciones, " +
  "frases del estilo 'Let's verify', 'Wait', 'I hear', 'Escucho', 'Verifico', " +
  "'Let me check', 'Let me re-listen', 'He says', 'He spells', 'Audio contents', " +
  "'Let's transcribe verbatim', 'Let's write', prefijos tipo 'Transcripción:', " +
  "markdown, viñetas, timecodes o etiquetas de hablante.\n" +
  "Si hay fragmentos inaudibles, márcalos con [inaudible] dentro del bloque.\n" +
  "Ejemplo de output válido:\n<<T>>¿Qué droga facilita la adhesión a GABA? 1. Benzodiazepinas. 2. Ansiolíticos. 3. Antipsicóticos.<</T>>";

/**
 * Saca el contenido entre los marcadores `<<T>>` y `<</T>>`.
 * Si no aparecen, devuelve el texto completo (fallback).
 */
function extractTranscriptBlock(text: string): string | null {
  const m = text.match(/<<T>>([\s\S]*?)<<\/T>>/);
  return m ? m[1].trim() : null;
}

/**
 * Filtro defensivo de la transcripción: descarta líneas que parecen
 * razonamiento del modelo ("Let's...", "Wait...", "He says...", etc.)
 * y se queda con el contenido limpio. Se aplica DESPUÉS de `sanitizeResponseText`.
 *
 * Si el prompt se cumplió y el output viene limpio, devuelve el texto tal cual.
 * Si el modelo igual filtró ruido, intenta reconstruir la pregunta real
 * descartando las líneas de "thinking".
 */
export function cleanTranscript(raw: string): string {
  if (!raw) return raw;

  // 1) Si el modelo respetó los marcadores `<<T>>`, usar eso directamente.
  const block = extractTranscriptBlock(raw);
  const source = block ?? raw;

  const lines = source
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length <= 1) return source.trim();

  // 2) Patrones que delatan razonamiento verbal del modelo. Si la línea
  //    empieza con alguno de estos prefijos (en ES o EN), la descartamos.
  const reasoningPrefixes = [
    "let's", "let me", "now,", "now ", "wait", "wait,",
    "first,", "ok,", "okay,", "yes,", "sure,",
    "listen", "i hear", "i need", "i should",
    "the user", "audio contents", "audio:",
    "verify", "verifico", "escucho", "verific",
    "he says", "he spells", "he reads", "he literally",
    "she says", "she spells",
    "let's transcribe", "let's verify", "let's check",
    "let's write", "let's listen", "let's carefully",
    "let's double", "let's re", "let's re-listen",
    "transcripción:", "transcripcion:", "respuesta:",
    "carefully", "double check", "double-check",
    "i'll", "i will",
  ];

  const isReasoning = (line: string): boolean => {
    const lower = line.toLowerCase();
    return reasoningPrefixes.some((p) => lower.startsWith(p));
  };

  const cleaned = lines.filter((l) => !isReasoning(l));

  if (cleaned.length === 0) {
    // Nada sobrevivió: devolvemos el bloque original como último recurso.
    return source.trim();
  }

  // 3) Si después de filtrar todavía quedan varias líneas, preferir las que
  //    parezcan pregunta real (empiezan con ¿, o contienen ?  cerca del final,
  //    o empiezan con mayúscula + verbo interrogativo típico, o listan opciones
  //    numeradas tipo "1. X. 2. Y.").
  const looksLikeQuestion = (line: string): boolean => {
    if (line.startsWith("¿")) return true;
    if (/\?\s*(\d+\.|[\s"])/.test(line)) return true;
    if (/^\d+\.\s+\S/.test(line)) return true;
    if (/^[A-ZÁÉÍÓÚÑ][^.]*\?/.test(line)) return true;
    return false;
  };

  const questionLines = cleaned.filter(looksLikeQuestion);
  if (questionLines.length > 0) {
    return questionLines.join(" ").replace(/\s+/g, " ").trim();
  }

  // 4) Si ninguna línea parece pregunta, devolver la línea más larga
  //    (suele ser la transcripción verbatim).
  return cleaned.reduce((a, b) => (b.length > a.length ? b : a), "").trim();
}

export async function transcribeAudio(
  base64Audio: string,
  mimeType: string,
  apiKey: string
): Promise<string> {
  const cleanKey = apiKey.trim();
  if (!cleanKey || cleanKey === "TU_API_KEY_AQUI") {
    throw new Error("Sin API Key para transcribir.");
  }
  const ai = new GoogleGenAI({ apiKey: cleanKey });
  const response = await ai.models.generateContent({
    model: GEMINI_MODEL,
    contents: [
      {
        parts: [
          { inlineData: { mimeType, data: base64Audio } },
          { text: TRANSCRIBE_INSTRUCTION },
        ],
      },
    ],
    config: {
      maxOutputTokens: 600,
      temperature: 0,
      thinkingConfig: { thinkingLevel: ThinkingLevel.MINIMAL },
    },
  });
  const sanitized = sanitizeResponseText((response?.text ?? "").trim());
  if (!sanitized) {
    throw new Error("Transcripción vacía.");
  }
  const cleaned = cleanTranscript(sanitized);
  return cleaned;
}

/** Cuenta palabras separadas por espacios (igual criterio que la UI). */
export function countWords(text: string): number {
  const t = (text ?? "").trim();
  if (!t) return 0;
  return t.split(/\s+/).length;
}

/**
 * Ampliación automática: si la respuesta salió por debajo del mínimo
 * (el modelo a veces ignora la extensión pedida), se le reenvía su propio
 * texto con los PDFs y se le pide desarrollarlo hasta 200-250 palabras,
 * en el mismo tono y formato. Se llama UNA sola vez por consulta, en
 * segundo plano dentro del flujo de "processing" (sin interacción).
 */
export async function expandAnswer(
  previousAnswer: string,
  apiKey: string,
  onProgress?: (msg: string) => void
): Promise<string> {
  const cleanKey = apiKey.trim();
  if (!cleanKey || cleanKey === "TU_API_KEY_AQUI") {
    throw new Error("Sin API Key para ampliar.");
  }
  const ai = new GoogleGenAI({ apiKey: cleanKey });
  onProgress?.("Ampliando respuesta…");
  const pdfParts = await buildKnowledgeBaseParts(ai, onProgress);
  const response = await ai.models.generateContent({
    model: GEMINI_MODEL,
    contents: [
      {
        parts: [
          ...pdfParts,
          {
            text:
              "Esta fue tu respuesta, pero quedó por debajo de las 200 palabras mínimas y eso es INACEPTABLE. " +
              "PROHIBIDO devolver menos de 200 palabras. Desarrollala hasta alcanzar entre 200 y 250 palabras, " +
              "manteniendo prosa continua, sin saludos, sin listas, sin cuadros y sin usar el término 'adaptación'. " +
              "Estrategia obligatoria: agregá al menos dos párrafos nuevos con precisiones teóricas de los PDFs " +
              "(citas de autor, categorías) y ejemplos fílmicos concretos con análisis de procedimientos formales. " +
              "Devolvé la respuesta COMPLETA ampliada, no solo lo agregado:\n\n" +
              previousAnswer,
          },
        ],
      },
    ],
    config: {
      systemInstruction: SYSTEM_PROMPT,
      maxOutputTokens: 2400,
      thinkingConfig: { thinkingLevel: ThinkingLevel.LOW },
      temperature: 0.3,
    },
  });
  const text = sanitizeResponseText((response?.text ?? "").trim());
  if (!text) {
    throw new Error("Ampliación vacía.");
  }
  return text;
}
