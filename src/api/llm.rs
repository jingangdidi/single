use openai_dive::v1::{
    api::Client,
    resources::chat::{
        ChatCompletionParametersBuilder,
        ChatCompletionResponseFormat,
        ChatMessage,
        ChatMessageContent,
    },
};
use serde_json::json;

use crate::{
    parse_paras::PARAS,
    openai::for_chat::not_use_stream, // 非stream，接收openai的完整答案
    error::MyError,
    tools::run_tools,
};

pub async fn run_llm() -> Result<(), MyError> {
    // 使用api key初始化
    let mut client = Client::new(PARAS.selected_model.api_key.clone());
    client.set_base_url(&PARAS.selected_model.endpoint); // 从0.7.0开始舍弃了new_with_base
    // 准备参数
    /*
    let parameters = ChatCompletionParametersBuilder::default()
        .model(model.clone()) // 指定模型，例如：Gpt4Engine::Gpt4O.to_string()
        .messages(get_messages(&uuid))
        .reasoning_effort(ReasoningEffort::Low) // 设置使用思维链，Low, Medium, High
        .response_format(ChatCompletionResponseFormat::Text)
        //.stream(stream) // 这里不需要设置，调用`create_stream`时会设置
        .build().map_err(|e| MyError::ChatCompletionError{error: e})?;
    */
    let mut para_builder = ChatCompletionParametersBuilder::default();
    para_builder.model(PARAS.selected_model.name.clone()); // 指定模型，例如：Gpt4Engine::Gpt4O.to_string()
    para_builder.response_format(ChatCompletionResponseFormat::Text);
    //para_builder.stream(stream); // 这里不需要设置，调用`create_stream`时会设置
    let lowercase_model = PARAS.selected_model.name.to_lowercase();
    if PARAS.selected_model.reasoning { // 对思维链模型设置effort
        para_builder.reasoning_effort(PARAS.selected_model.effort.clone()); // 设置使用思维链，Low（思考的少，简单问答）, Medium（思考适中，多步骤推理）, High（思考更多，复杂逻辑推导）
        // 开启思考，不同模型思考的设置不同
        if lowercase_model.starts_with("deepseek") {
            // deepseek: https://api-docs.deepseek.com/
            para_builder.extra_body(json!({"thinking": {"type": "enabled"}}));
        } else if lowercase_model.starts_with("qwen") {
            // Qwen: https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions#05cfceb898csa
            para_builder.extra_body(json!({"enable_thinking": true}));
        } else if lowercase_model.starts_with("kimi") {
            // kimi: https://platform.kimi.com/docs/api/models-overview
            para_builder.extra_body(json!({"thinking": {"type": "enabled"}}));
        } else if lowercase_model.starts_with("glm") {
            // glm: https://docs.bigmodel.cn/cn/guide/develop/openai/introduction
            para_builder.extra_body(json!({"thinking": {"type": "enabled"}}));
        } else if lowercase_model.starts_with("minimax") {
            // minimax: https://platform.minimaxi.com/docs/api-reference/text-openai-api
            // 目前不支持关闭thinking：https://github.com/MiniMax-AI/MiniMax-M2/issues/68
            //para_builder.extra_body(json!({"reasoning_split": true}));
        }
    } else {
        // 关闭思考，不同模型思考的设置不同
        if lowercase_model.starts_with("deepseek") {
            // deepseek: https://api-docs.deepseek.com/
            para_builder.extra_body(json!({"thinking": {"type": "disabled"}}));
        } else if lowercase_model.starts_with("qwen") {
            if PARAS.selected_model.endpoint.starts_with("http://") { // local model
                // https://modelscope.cn/models/Qwen/Qwen3.5-397B-A17B
                // https://modelscope.cn/models/Qwen/Qwen3.6-35B-A3B#instruct-or-non-thinking-mode
                para_builder.extra_body(json!({"chat_template_kwargs": {"enable_thinking": false}}));
            } else {
                // Qwen: https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions#05cfceb898csa
                para_builder.extra_body(json!({"enable_thinking": false}));
            }
        } else if lowercase_model.starts_with("kimi") {
            // kimi: https://platform.kimi.com/docs/api/models-overview
            para_builder.extra_body(json!({"thinking": {"type": "disabled"}}));
        } else if lowercase_model.starts_with("glm") {
            // glm: https://docs.bigmodel.cn/cn/guide/develop/openai/introduction
            para_builder.extra_body(json!({"thinking": {"type": "disabled"}}));
        } else if lowercase_model.starts_with("minimax") {
            // minimax，目前不支持关闭thinking：https://github.com/MiniMax-AI/MiniMax-M2/issues/68
            //para_builder.extra_body(json!({"reasoning_split": false}));
        }
    }
    if let Some(temp) = PARAS.selected_model.temperature {
        para_builder.temperature(temp);
    }
    if let Some(top_p) = PARAS.selected_model.top_p {
        para_builder.top_p(top_p);
    }
    let mut messages = Vec::new();
    // 插入 prompt
    if let Some(p) = &PARAS.selected_prompt {
        messages.push(
            ChatMessage::User{
                content: ChatMessageContent::Text(p.to_string()),
                name: None,
            }
        );
    }
    // 插入问题
    messages.push(
        ChatMessage::User{
            content: ChatMessageContent::Text(PARAS.question.clone()),
            name: None,
        }
    );
    // 提问
    if PARAS.selected_tools.is_some() || PARAS.selected_skills.is_some() {
        let result = run_tools(PARAS.selected_tools.clone(), PARAS.selected_skills.clone(), client, para_builder.clone(), &PARAS.selected_model.name, messages).await?;
        println!("{}", result);
    } else {
        para_builder.messages(messages);
        let parameters = para_builder.build().map_err(|e| MyError::ChatCompletionError{error: e})?;
        let (whole_answer, _thinking_content) = not_use_stream(client, parameters, &PARAS.selected_model.name).await?;
        println!("{}", whole_answer);
    }
    Ok(())
}
