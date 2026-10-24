# ESTRUCTURA DEL PROYECTO

## TÍTULO DEL PROYECTO

Pronóstico semanal de casos de dengue en Colombia mediante aprendizaje automático (Weekly Dengue Case Forecasting in Colombia)

## PLANTEAMIENTO DEL PROBLEMA Y JUSTIFICACIÓN

El dengue es uno de los problemas de salud pública más relevantes de Colombia: los brotes regresan año tras año, saturan hospitales y servicios de salud, sobre todo en municipios cálidos y de baja altitud. Hoy la respuesta es **reactiva**: los casos se cuentan cuando ya ocurrieron, los recursos llegan cuando el pico ya comenzó y las campañas de prevención se lanzan tarde, cuando el brote es mucho más difícil de contener.

**Pregunta de investigación:** ¿es posible anticipar con algunas semanas de antelación el número de casos semanales de dengue a partir de los registros históricos del SIVIGILA y de variables climáticas (precipitación y temperatura del IDEAM)?

**Justificación:** un pronóstico semanal permite planificar personal, camas e insumos y activar el control vectorial y la prevención a tiempo. Los datos son abiertos (datos.gov.co), el clima influye directamente en la reproducción del mosquito y en la transmisión, y las técnicas de series de tiempo y aprendizaje automático permiten convertir esos registros en una herramienta de alerta temprana sencilla y reproducible.

## OBJETIVOS DEL PROYECTO

### OBJETIVO GENERAL

Desarrollar y comparar modelos de aprendizaje automático y series de tiempo que pronostiquen los casos semanales de dengue en municipios de Colombia, y ponerlos a disposición en un tablero web de alerta temprana.

### OBJETIVOS ESPECÍFICOS

1. Descargar, limpiar y unificar los registros de dengue del SIVIGILA (datos.gov.co) de municipios como Bucaramanga, Tuluá y Pereira, alineando columnas entre fuentes.
2. Construir la serie semanal (por semana epidemiológica y comuna) y agregar variables rezagadas de casos, precipitación y temperatura (IDEAM).
3. Entrenar y comparar un modelo base, ARIMA, Prophet y modelos de ML con rezagos (Random Forest / XGBoost), exigiendo que superen al modelo base.
4. Evaluar con partición temporal (entrenamiento: semanas anteriores; prueba: últimas semanas) usando MAE, RMSE, detección de picos y gráficos real vs. predicho.
5. Identificar qué rezagos climáticos son más informativos y desplegar un tablero web (Render) que muestre pronósticos y alertas.

## MARCO REFERENCIAL (Opcional)

- **Epidemiológico:** dengue como enfermedad transmitida por vectores (*Aedes aegypti*); vigilancia en salud pública con SIVIGILA (INS) y semanas epidemiológicas.
- **Climático:** la lluvia y la temperatura afectan el ciclo del mosquito; sus efectos aparecen con retraso (p. ej., la lluvia de hace 3 semanas puede explicar los casos de hoy).
- **Técnico:** series de tiempo (ARIMA, Prophet), aprendizaje supervisado con rezagos (Random Forest, XGBoost), validación temporal y métricas MAE/RMSE.
- **Datos:** datos.gov.co (SIVIGILA, conjuntos municipales) e IDEAM (datos diarios por estación).

## METODOLOGÍA A UTILIZAR Y CRONOGRAMA DE DESARROLLO

**Metodología**

1. **Datos:** descarga vía API Socrata, estandarización de columnas por municipio y conversión del clima diario a semanas epidemiológicas.
2. **Series y características:** conteo de casos por semana (y comuna) y variables de rezago de casos, lluvia y temperatura.
3. **Modelado:** modelo base (semana siguiente = semana actual o promedio histórico de esa semana), ARIMA, Prophet y Random Forest/XGBoost.
4. **Evaluación:** partición por tiempo (nunca aleatoria) para evitar resultados optimistas; MAE, RMSE, detección de picos y gráficos real vs. predicho.
5. **Entrega:** tablero web desplegado en Render, con código reproducible (`src/dengue`, pruebas con `pytest`).

**Cronograma (16 semanas)**

| Semanas | Actividad |
|---------|-----------|
| 1–3 | Descarga y limpieza de los conjuntos de datos |
| 4–6 | Construcción de la serie semanal y variables de rezago |
| 7–11 | Entrenamiento y comparación de modelos |
| 12–14 | Evaluación, análisis de rezagos climáticos y de picos |
| 15–16 | Tablero web en Render, documento síntesis y presentación final |

**Riesgos:** series cortas (poco historial estacional), formatos distintos entre municipios, unión de clima diario por estación con semanas epidemiológicas, y validación cuidadosa para no sobreestimar el desempeño.

## REFERENCIAS BIBLIOGRÁFICAS (opcional)

- Instituto Nacional de Salud. *Sistema de Vigilancia en Salud Pública – SIVIGILA*. Colombia.
- Datos Abiertos Colombia (datos.gov.co). Conjuntos de datos de vigilancia de dengue.
- IDEAM. Datos hidrometeorológicos (precipitación y temperatura).
- Hyndman, R. J., & Athanasopoulos, G. *Forecasting: Principles and Practice* (3.ª ed.). OTexts.
- Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American Statistician*, 72(1).
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1).
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*.

---

## PRODUCTOS ESPERADOS (marque con una X)

| X | Producto |
|---|----------|
| X | Documento síntesis de PGC |
| X | Software |
|   | Sistema de información |
| X | Prototipo |
|   | Artículo científico |
|   | Póster- Ponencia |
|   | Comunidad de práctica y redes de conocimiento |
|   | Evento científico o académico |
|   | Ediciones de revista o libro de divulgación |
|   | Producción de contenidos digitales |
|   | Publicaciones editoriales no especializadas |
| X | Desarrollo Web |
|   | Creación de cursos |
|   | Emprendimiento e innovación |
|   | Otro ¿Cuál? |
