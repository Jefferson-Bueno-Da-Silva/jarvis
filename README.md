# Jarvis - Agente de Google Tasks com LangGraph

Agente em Python que usa **Gemini (Google Generative AI)** com **tool calling** para executar operações no **Google Tasks**.

O fluxo principal:
1. Recebe uma instrução em linguagem natural.
2. O modelo decide se precisa chamar uma ferramenta.
3. As ferramentas executam operações no Google Tasks (CRUD).
4. O agente responde em português com base no resultado.

## Requisitos

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) instalado
- Conta Google Cloud com API Google Tasks habilitada
- Arquivo `credentials.json` (OAuth client) na raiz do projeto

## Instalação

### 1. Clonar e entrar no projeto

```bash
git clone <url-do-repo>
cd jarvis
```

### 2. Instalar dependências

```bash
uv sync
```

## Configuração de ambiente

Crie/edite o arquivo `.env` na raiz com:

```env
GOOGLE_API_KEY=<sua_chave_gemini>
LANGFUSE_PUBLIC_KEY=<opcional>
LANGFUSE_SECRET_KEY=<opcional>
LANGFUSE_BASE_URL=<opcional>
```

Observações:
- `GOOGLE_API_KEY` é necessária para o modelo Gemini.
- Variáveis `LANGFUSE_*` são usadas para observabilidade/tracing.

## Configuração Google Tasks (OAuth)

1. No Google Cloud, habilite a **Google Tasks API**.
2. Crie credenciais OAuth de aplicativo desktop.
3. Baixe o JSON e salve como `credentials.json` na raiz.
4. Na primeira execução, o app abrirá o login OAuth e criará `token.json` automaticamente.

Escopo usado:
- `https://www.googleapis.com/auth/tasks`

## Como executar

```bash
uv run python main.py
```

O script principal chama `run_pipeline()` e imprime as mensagens finais do agente.

Também é possível passar a mensagem direto por argumento:

```bash
uv run python main.py "Liste as minhas tarefas"
```

## API HTTP (pasta `APP`)

A API foi criada em `APP/main.py` usando FastAPI para servir o agente via HTTP.

### Subir servidor

```bash
uv run uvicorn APP.main:app --reload
```

### Endpoints

- `GET /health`  
Retorna status da API.

- `POST /agent`  
Envia uma mensagem para o agente.

Payload:

```json
{
  "message": "Liste as minhas tarefas"
}
```

Resposta:

```json
{
  "answer": "Aqui estão as suas tarefas...",
  "llm_calls": 2
}
```

Swagger UI:
- `http://127.0.0.1:8000/docs`

## Organização dos arquivos

- `main.py`  
Ponto de entrada CLI da aplicação. Executa o pipeline e exibe a saída final.

- `APP/main.py`  
Servidor HTTP com FastAPI para consumir o agente pelo protocolo HTTP.

- `src/agent/main.py`  
Orquestra o agente com `LangGraph`.
  - Define o `SystemMessage`.
  - Faz loop entre `llm_call` e `tool_node`.
  - Decide continuidade com `should_continue`.
  - Compila e executa o grafo.

- `src/models/model.py`  
Configura o modelo `ChatGoogleGenerativeAI` (`gemini-2.5-flash`) e cria o mapa de ferramentas (`tools_by_name`).

- `src/tools/tools.py`  
Define ferramentas do agente (decoradas com `@tool`) e schemas (`pydantic`) para validação de entrada.

- `src/services/GoogleTasks/googleTask.py`  
Cliente de integração com Google Tasks API.
  - autenticação OAuth (`credentials.json` + `token.json`)
  - métodos de listar, criar, atualizar e deletar tarefas

- `src/static/graph_xray.png`  
Imagem do grafo do agente (artefato estático).

- `pyproject.toml` / `uv.lock`  
Metadados do projeto e travamento de dependências.

## Features utilizadas (e por quê)

- **Google Tasks (CRUD)**  
Permite ao agente executar operações reais de tarefas: listar, criar, atualizar e apagar.

- **Tool Calling com LangChain**  
Transforma funções Python em ferramentas que o LLM pode invocar de forma estruturada.

- **LangGraph**  
Controla o ciclo de decisão do agente (`LLM -> Tool -> LLM`) até concluir a resposta.

- **Gemini (`gemini-2.5-flash`)**  
Modelo responsável por interpretar a solicitação e decidir quais ferramentas usar.

- **Pydantic nos inputs das tools**  
Valida argumentos (ex.: campos obrigatórios, limites, formatos), reduzindo erros de execução.

- **Langfuse**  
Registra observabilidade (chamadas de LLM e tools), útil para debugging e monitoramento.

- **dotenv**  
Carrega variáveis de ambiente de forma simples via `.env`.

## Exemplo de uso (comportamento esperado)

Pedidos como:
- "Liste as minhas tarefas"
- "Crie uma tarefa 'Comprar leite' para 2026-02-26T18:00:00.000Z"
- "Atualize a tarefa X para completed"
- "Apague a tarefa Y"

O agente usa as ferramentas de Google Tasks para executar a ação e responde em português.

## Problemas comuns

- `credentials.json` ausente: sem ele o OAuth não inicia.
- `GOOGLE_API_KEY` ausente/inválida: o modelo Gemini não responde.
- `token.json` com escopo antigo: apague `token.json` e autentique novamente.
