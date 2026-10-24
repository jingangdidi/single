use openai_dive::v1::{
    api::Client,
    resources::{
        chat::{
            ChatCompletionParameters,
            ChatMessage,
            ChatMessageContent,
        },
        shared::Usage,
    },
};
//use tokio::time::{sleep, Duration};

use crate::error::MyError;

const THOUGHT_START: &[&str] = &["<think>", "<thought>", "<thinking>", "<reasoning>"];
const THOUGHT_END: &[&str] = &["</think>", "</thought>", "</thinking>", "</reasoning>"];

/// 非stream，接收openai的完整答案，并将完整答案记录到该uuid中，最后返回完整答案以及思考内容
/// uuid: 当前对话的uuid
/// client: 创建的openai客户端
/// parameters: 提问的参数
pub async fn not_use_stream(
    client: Client,
    parameters: ChatCompletionParameters,
) -> Result<(String, Option<String>), MyError> {
    //let result = client.chat().create(parameters).await.map_err(|e| MyError::ApiError{uuid: uuid.clone(), error: e})?;
    let mut whole_answer = "".to_string(); // 存储完整答案
    let mut thinking = "".to_string(); // 思考部分
    match client.chat().create(parameters).await { // 这里遇到错误不能直接返回，否则服务端与前端id差一个，后面代码insert_message没有执行，下个问题会显示在这个未回答完的答案末尾
        Ok(result) => {
            // 获取答案
            match &result.choices[0].message {
                ChatMessage::System{content, ..} => {
                    match content {
                        ChatMessageContent::Text(res) => {
                            whole_answer = res.to_string();
                            //println!("{}", res);
                        },
                        ChatMessageContent::ContentPart(res_vec) => eprintln!("System ChatMessageContent::ContentPart {:?}", res_vec),
                        ChatMessageContent::None => eprintln!("System ChatMessageContent::None"),
                    }
                },
                ChatMessage::User{content, ..} => {
                    match content {
                        ChatMessageContent::Text(res) => {
                            whole_answer = res.to_string();
                            //println!("{}", res);
                        },
                        ChatMessageContent::ContentPart(res_vec) => eprintln!("System ChatMessageContent::ContentPart {:?}", res_vec),
                        ChatMessageContent::None => eprintln!("System ChatMessageContent::None"),
                    }
                },
                ChatMessage::Assistant{content: tmp_content, reasoning: tmp_reasoning_content, reasoning_content: tmp_reasoning_content_deepseek, ..} => {
                    if let Some(c) = tmp_reasoning_content {
                        thinking = format!("<div class='think'>thinking ...<br>{}srxtzn</div>srxtznsrxtzn---srxtznsrxtzn", c);
                    } else if let Some(c) = tmp_reasoning_content_deepseek {
                        thinking = format!("<div class='think'>thinking ...<br>{}srxtzn</div>srxtznsrxtzn---srxtznsrxtzn", c);
                    }
                    match tmp_content {
                        Some(c) => {
                            match c {
                                ChatMessageContent::Text(res) => {
                                    // 如果思考部分没有单独方在`reasoning_content`，则尝试从答案主体中获取
                                    if thinking.is_empty() {
                                        for (start, end) in THOUGHT_START.iter().zip(THOUGHT_END.iter()) {
                                            if res.starts_with(start) && res.contains(end) {
                                                let (first, second) = res.split_once(end).unwrap();
                                                thinking = format!("<div class='think'>thinking ...<br>{}srxtzn</div>srxtznsrxtzn---srxtznsrxtzn", first);
                                                whole_answer = second.to_string();
                                                break
                                            }
                                        }
                                        if thinking.is_empty() {
                                            whole_answer = res.to_string();
                                        }
                                    } else {
                                        whole_answer = res.to_string();
                                    }
                                    //println!("{}", res);
                                },
                                ChatMessageContent::ContentPart(res_vec) => eprintln!("System ChatMessageContent::ContentPart {:?}", res_vec),
                                ChatMessageContent::None => eprintln!("System ChatMessageContent::None"),
                            }
                        },
                        None => eprintln!("Assistant: None"), // println!("Assistant: None"),
                    }
                },
                ChatMessage::Developer{content, ..} => {
                    match content {
                        ChatMessageContent::Text(res) => {
                            whole_answer = res.to_string();
                            //println!("{}", res);
                        },
                        ChatMessageContent::ContentPart(res_vec) => eprintln!("System ChatMessageContent::ContentPart {:?}", res_vec),
                        ChatMessageContent::None => eprintln!("System ChatMessageContent::None"),
                    }
                },
                ChatMessage::Tool{content, ..} => eprintln!("{}", content), // println!("{}", content),
            }
            if let Some(usage) = result.usage {
                get_print_token(usage);
            }
        },
        Err(e) => println!("{:?}", e),
    }
    Ok((whole_answer, if thinking.is_empty() { None } else { Some(thinking) }))
}

// print and update total token usage
pub fn get_print_token(usage: Usage) -> Option<(u32, u32, u32)> {
    // 缓存命中的token数
    let cached_tokens = match usage.prompt_tokens_details {
        Some(details) => details.cached_tokens.unwrap_or(0),
        None => 0,
    };
    // 推理部分的token数
    let reasoning_tokens = match usage.completion_tokens_details {
        Some(details) => details.reasoning_tokens.unwrap_or(0),
        None => 0,
    };
    match (usage.prompt_tokens, usage.completion_tokens) {
        (Some(prompt_tokens), Some(completion_tokens)) => {
            eprintln!("{{prompt_tokens: {} (cached_tokens: {}), completion_tokens: {} (reasoning_tokens: {}), total_tokens: {}}}", prompt_tokens, cached_tokens, completion_tokens, reasoning_tokens, usage.total_tokens);
            Some((prompt_tokens, completion_tokens, usage.total_tokens))
        },
        (Some(prompt_tokens), None) => {
            eprintln!("{{prompt_tokens: {} (cached_tokens: {}), completion_tokens: None, total_tokens: {}}}", cached_tokens, prompt_tokens, usage.total_tokens);
            Some((prompt_tokens, 0, usage.total_tokens))
        },
        (None, Some(completion_tokens)) => {
            eprintln!("{{prompt_tokens: None, completion_tokens: {} (reasoning_tokens: {}), total_tokens: {}}}", completion_tokens, reasoning_tokens, usage.total_tokens);
            Some((0, completion_tokens, usage.total_tokens))
        },
        (None, None) => {
            eprintln!("{{prompt_tokens: None, completion_tokens: None, total_tokens: {}}}", usage.total_tokens);
            Some((0, 0, usage.total_tokens))
        },
    }
}
