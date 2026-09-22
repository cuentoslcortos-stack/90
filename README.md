## 90A — tutor académico por voz

Tutor IA por voz para la materia **Biología del Comportamiento**
(Cátedra Muzio, Psicología UBA), con paleta **verde**.
Toda la funcionalidad (grabación por voz, envío a Gemini, lectura
TTS, guardado automático de Q&A en GitHub) queda intacta.

### Pendiente para terminar la ingesta

| Qué                          | Dónde                                                        |
| ---------------------------- | ------------------------------------------------------------ |
| Etiqueta visible             | `ASSISTANT_LABEL` en `src/lib/gemini.ts`                     |
| Prompt del sistema           | `SYSTEM_PROMPT` en `src/lib/gemini.ts`                       |
| PDFs de la base de conocimiento | `PDF_SOURCES` en `src/lib/gemini.ts` + archivos en `public/` |
| Modelo de IA                 | `GEMINI_MODEL` en `src/lib/gemini.ts` (default `gemini-3.6-flash`) |
| Volumen de los sonidos       | `SOUND_VOLUME` en `src/lib/sounds.ts`                        |