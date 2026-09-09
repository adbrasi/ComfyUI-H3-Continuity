# Ajuste automático da resolução na montagem

Continuação e RETAKE agora redimensionam source_images para a resolução dos frames gerados, com Lanczos e recorte central, quando necessário. É a mesma política usada no Import e no Prepare da continuação. Com proporções diferentes, as bordas são recortadas; não há esticamento da imagem.

A quantidade e a ordem dos frames, a duração e as amostras do áudio original são preservadas. Entradas na mesma resolução seguem o caminho anterior, sem reamostragem. No RETAKE, a preservação fora do intervalo editado se refere aos pixels da fonte ajustada à resolução de saída.

Validação: 58 testes aprovados. Os dois novos casos cobrem resoluções e proporções diferentes, recorte central, ordem dos frames, conservação do áudio e ausência de alterações nos tensores de entrada. Os checkpoints anteriores permanecem intactos.
