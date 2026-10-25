use std::collections::HashMap;
use std::path::Path;

use openai_dive::v1::{
    api::Client,
    resources::chat::{
        ChatCompletionFunction,
        ChatCompletionParametersBuilder,
        ChatCompletionTool,
        ChatCompletionToolType,
        ChatMessage,
        ChatMessageContent,
        ChatMessageContentPart,
        ChatMessageImageContentPart,
        ImageUrlType,
    },
};
//use regex::Regex;
use serde::{Deserialize, de::DeserializeOwned};
use serde_json::{json, Value};

use crate::{
    openai::for_tool::{
        call_tool_not_use_stream,
        CallToolResult,
    },
    parse_paras::PARAS,
    error::MyError,
    skills::SelectedSkills,
};

pub mod built_in_tools;
pub mod external_tools;

use built_in_tools::{
    BuiltInTools,
    filesystem::{
        read_file::Params as ReadFileParams,
        read_multiple_files::Params as ReadMultipleFilesParams,
        image_generation::image_generation,
        edit_image::edit_image,
    },
    deepseek_search::{
        search::deepseek_search,
        fetch::deepseek_fetch,
    },
};

#[cfg(feature = "screen-capture")]
use built_in_tools::filesystem::screen_capture::Params as ScreenCaptureParams;

use external_tools::ExternalTools;

/// html pulldown option selected tools
#[derive(Clone, Debug)]
pub enum SelectedTools {
    All,                   // all tools
    AllBuiltIn,            // all built-in tools
    AllExternal,           // all external tools
    Group(String),         // built-in group, start with `built_in_group_`, select all tools of one group
    Single(String),        // single built-in or external tool id
    AllMcp,                // all mcp tools
    Multiple(Vec<String>), // multiple tools
    McpServer(String),     // single mcp server id, start with `mcp_server_`, select all tools of one server
    McpTool(String),       // single mcp tool `name__id`, select by tool name and server id
}

/// trait for built-in tools & external tools
pub trait MyTools {
    /// run tool
    fn run(&self, id: &str, args: &str) -> Result<(String, Option<String>), MyError>;

    /// get all selected tools name (name format: `name__id`, max name length is 26), description and schema
    fn get_desc_and_schema(&self, selected_tools: Vec<String>) -> Vec<(String, String, Value)>;

    /// select all tools, return uuid vector
    fn select_all_tools(&self) -> Vec<String>;

    /// get all tools info, return vector of (uuid, name, description)
    fn get_all_tools_info(&self) -> Vec<(String, String, String)>;

    /// select tools by uuid first part, return uuid vector
    fn select_multiple_tools(&self, ids: &Vec<String>) -> Vec<String>;
}

/// all tools: built-in tools & external tools
pub struct Tools {
    built_in: BuiltInTools,
    external: ExternalTools,
}

impl Tools {
    /// create new Tools, ExternalTools from config file
    pub fn new(external: ExternalTools) -> Result<Self, MyError> {
        let built_in = BuiltInTools::new()?;
        Ok(Self {built_in, external})
    }

    /// run tool
    pub fn run(&self, id: &str, args: &str) -> Result<(String, Option<String>), MyError> {
        if self.built_in.id_map.contains_key(id) {
            self.built_in.run(id, args)
        } else if self.external.id_map.contains_key(id) {
            self.external.run(id, args)
        } else {
            Err(MyError::ToolNotExistError{id: id.to_string(), info: "Tools::run()".to_string()})
        }
    }

    /// get all selected tools for LLM api function calling
    /// tool name format: `name__id`, max name length is 26
    pub fn get_desc_and_schema(&self, selected_tools: &Option<SelectedTools>) -> Result<Vec<ChatCompletionTool>, MyError> {
        // get selected tools
        let (selected_builtin_tools, selected_external_tools) = match selected_tools {
            Some(s) => match s {
                SelectedTools::All => (self.built_in.select_all_tools(), self.external.select_all_tools()), // all tools
                SelectedTools::AllBuiltIn => (self.built_in.select_all_tools(), Vec::new()), // all built-in tools
                SelectedTools::AllExternal => (Vec::new(), self.external.select_all_tools()), // all external tools
                SelectedTools::Group(group) => (self.built_in.select_tools_by_group(&group)?, Vec::new()), // built-in group
                SelectedTools::Single(id) => { // single built-in or external tool id
                    if self.built_in.id_map.contains_key(id) {
                        (vec![id.to_string()], Vec::new())
                    } else if self.external.id_map.contains_key(id) {
                        (Vec::new(), vec![id.to_string()])
                    } else {
                        return Err(MyError::ToolNotExistError{id: id.to_string(), info: "Tools::select_tools()".to_string()})
                    }
                },
                SelectedTools::AllMcp => (Vec::new(), Vec::new()), // all mcp tools
                SelectedTools::Multiple(multiple) => (self.built_in.select_multiple_tools(&multiple), self.external.select_multiple_tools(&multiple)),
                SelectedTools::McpServer(_) => (Vec::new(), Vec::new()), // single mcp server id, start with `mcp_server_`, select all tools of one server
                SelectedTools::McpTool(_) => (Vec::new(), Vec::new()), // single mcp tool `name__id`, select by tool name and server id
            },
            None => (Vec::new(), Vec::new()), // not select any tool
        };
        // get name, description and schema
        let mut tools: Vec<ChatCompletionTool> = Vec::new();
        let desc_and_schema_builtin = self.built_in.get_desc_and_schema(selected_builtin_tools);
        let desc_and_schema_external = self.external.get_desc_and_schema(selected_external_tools);
        for tool in desc_and_schema_builtin.into_iter().chain(desc_and_schema_external) {
            tools.push(
                ChatCompletionTool {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionFunction {
                        name: tool.0,
                        description: Some(tool.1),
                        parameters: tool.2,
                    },
                }
            );
        }
        Ok(tools)
    }

    /// get all tools info, return vector of (uuid, name, description)
    pub fn get_all_tools_info(&self) -> Vec<(String, String, String)> {
        let mut tools = PARAS.tools.built_in.get_all_tools_info();
        let mut external_tools = PARAS.tools.external.get_all_tools_info();
        tools.append(&mut external_tools);
        tools
    }

    /// check contain tool id
    pub fn contain_tool_id(&self, id: &str) -> bool {
        if self.built_in.id_map.contains_key(id) {
            true
        } else if self.external.id_map.contains_key(id) {
            true
        } else {
            false
        }
    }

    /// get tool id by name
    pub fn get_tool_id_by_name(&self, name: &str) -> Option<String> {
        // 从 built-in 工具中查找
        for (k, v) in &self.built_in.id_map {
            if v.tool.name() == name {
                return Some(k.clone())
            }
        }
        // 从 external 工具中查找
        for (k, v) in &self.external.id_map {
            if v.name == name {
                return Some(k.clone())
            }
        }
        None
    }
}

/// chat with tools: built-in + external tools
/// This just a simple loop, if not call tool, break the loop, return result
/// +-------------------------------------------------------------+
/// |      user query                                             |
/// |          |                                                  |
/// | +--------+------------+                                     |
/// | |        |            |                                     |
/// | |        V            |                                     |
/// | | context messages <--+--------------------+                |
/// | |        |            |                    |                |
/// | |        V            |                    | add            |
/// | |       LLM           |                    | each           |
/// | |        |            |                    | call           |
/// | |        V            |                    | tool           |
/// | |    call tool ?      |                    | result         |
/// | |        |            |                    | to             |
/// | |    +-------+        |                    | context        |
/// | |    |       |        |                    |                |
/// | |    V       V        |    +------------------------------+ |
/// | |   No      Yes ------+--> | run MCP tools, return result | |
/// | |    |                |    +------------------------------+ |
/// | +----+----------------+                                     |
/// |      |                                                      |
/// |      V                                                      |
/// |   result                                                    |
/// +-------------------------------------------------------------+
/// function calling:
/// https://api-docs.deepseek.com/zh-cn/guides/function_calling
/// https://docs.bigmodel.cn/cn/guide/capabilities/function-calling
/// https://platform.moonshot.cn/docs/api/tool-use
pub async fn run_tools(
    selected_tools: Option<SelectedTools>,
    selected_skills: Option<SelectedSkills>,
    client: Client,
    mut para_builder: ChatCompletionParametersBuilder,
    model: &str,
    messages: Vec<ChatMessage>,
) -> Result<(String, Vec<ChatMessage>), MyError> {
    // get built-in and external tools schame
    let mut tool_schema = PARAS.tools.get_desc_and_schema(&selected_tools)?;
    // get mcp tools schema
    let mcp_schema = PARAS.mcp_servers.get_desc_and_schema(&selected_tools).await?;
    // 如果是自定义选择的多个工具，打印每隔工具名
    if let Some(SelectedTools::Multiple(_)) = selected_tools {
        let mut names = Vec::new();
        for t in &tool_schema {
            names.push(t.function.name.as_str());
        }
        for t in &mcp_schema {
            names.push(t.function.name.as_str());
        }
        eprintln!("selected multiple tools: {}", names.join(", "));
    }
    // 如果指定了skills，则加入一个激活指定skill的tool
    if selected_skills.is_some() {
        tool_schema.push(
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: ChatCompletionFunction {
                    name: "activate_skill".to_string(),
                    description: Some("Activate an agent skill to load its full instructions. IMPORTANT: You must CALL this tool (not write it as text) to activate a skill. Use this when you see a relevant skill in the available skills list and need its detailed instructions to complete a task.".to_string()),
                    parameters: json!({
                        "properties": {
                            "skill_name": {
                                "type": "string",
                                "description": "The name of the skill to activate.",
                            },
                        },
                        "required": ["skill_name"],
                        "type": "object",
                    }),
                },
            }
        );
        // 如果没有选择`run_command`，则自动添加这个工具
        if tool_schema.iter().any(|t| !t.function.name.starts_with("run_command")) {
            let run_command_schema: Vec<_> = PARAS
                .tools
                .built_in
                .id_map
                .iter()
                .filter(|(_, v)| v.tool.name() == "run_command")
                .map(|(k, v)| (format!("{}__{}", v.tool.name(), k), v.tool.description(), v.tool.schema()))
                .collect();
            tool_schema.push(
                ChatCompletionTool {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionFunction {
                        name: run_command_schema[0].0.clone(),
                        description: Some(run_command_schema[0].1.clone()),
                        parameters: run_command_schema[0].2.clone(),
                    },
                }
            );
        }
        // 如果没有选择`run_script`，则自动添加这个工具
        if tool_schema.iter().any(|t| !t.function.name.starts_with("run_script")) {
            let run_command_schema: Vec<_> = PARAS
                .tools
                .built_in
                .id_map
                .iter()
                .filter(|(_, v)| v.tool.name() == "run_script")
                .map(|(k, v)| (format!("{}__{}", v.tool.name(), k), v.tool.description(), v.tool.schema()))
                .collect();
            tool_schema.push(
                ChatCompletionTool {
                    r#type: ChatCompletionToolType::Function,
                    function: ChatCompletionFunction {
                        name: run_command_schema[0].0.clone(),
                        description: Some(run_command_schema[0].1.clone()),
                        parameters: run_command_schema[0].2.clone(),
                    },
                }
            );
        }
    }
    // 加上 sub_agent 工具
    {
        let sub_agent_schema: Vec<_> = PARAS
            .tools
            .built_in
            .id_map
            .iter()
            .filter(|(_, v)| v.tool.name() == "sub_agent")
            .map(|(k, v)| (format!("{}__{}", v.tool.name(), k), v.tool.description(), v.tool.schema()))
            .collect();
        tool_schema.push(
            ChatCompletionTool {
                r#type: ChatCompletionToolType::Function,
                function: ChatCompletionFunction {
                    name: sub_agent_schema[0].0.clone(),
                    description: Some(sub_agent_schema[0].1.clone()),
                    parameters: sub_agent_schema[0].2.clone(),
                },
            }
        );
    }
    // prepare buider for call tool
    tool_schema.extend(mcp_schema);
    // tool名称加了`_uuid第一部分`保证不重复，但是上下文很长时可能导致模型返回的要调用的模型丢失了`_uuid第一部分`，导致调用模型失败
    // 这里把不含`_uuid第一部分`的tool名称作为key，将对应的完整tool名称作为value，存入HashMap，当`_uuid第一部分`丢失，也能确认具体要调用的模型
    let tool_map = get_tool_hashmap(&tool_schema);
    let para_builder_for_sub_agent = para_builder.clone();
    para_builder.tools(tool_schema.clone());
    let mut history_messages = messages; // store all messages as context log, this is temp history, will not add to the user's main history
    if let Some(sele_skills) = selected_skills {
        let skill_prompt = match sele_skills {
            SelectedSkills::All => PARAS.skills.get_all_available_skills_prompt(),
            SelectedSkills::Group(group) => PARAS.skills.get_group_available_skills_prompt(group),
            SelectedSkills::Single(idx) => PARAS.skills.get_single_available_skill_prompt(idx),
        };
        history_messages.push(
            ChatMessage::User{
                content: ChatMessageContent::Text(skill_prompt),
                name: None,
            }
        );
    }
    let mut call_tool_count = 0; // call tool count
    let call_tool_limit = call_tool_count+100; // 调用工具的次数限制
    let mut try_count = 0;
    let mut is_first: bool;
    let mut real_name: String; // 要调用的工具的真实名称，含有`_uuid第一部分`后缀
    let final_result: String; // 最终结果

    //'outer: loop {
    loop {
        // send query to LLM
        para_builder.messages(history_messages.clone());
        let parameters = para_builder.build().map_err(|e| MyError::ChatCompletionError{error: e})?;
        let answer = call_tool_not_use_stream(client.clone(), parameters).await?;
        // if answer is call tool result, continue; else break
        match answer {
            CallToolResult::CallTool((raw_message, call_tool_result)) => { // (ChatMessage, Vec<(tool name, tool args, call tool id, content)>)
                is_first = true;
                for j in call_tool_result {
                    // call tool
                    let mut name_id: Vec<&str> = j.0.split("__").collect();
                    if name_id.len() < 2 && name_id[0] != "activate_skill" {
                        if let Some(tools_vec) = tool_map.get(&j.0) {
                            if tools_vec.len() > 1 {
                                eprintln!("call tool error, llm only return tool name prefix `{}`, but this prefix have multiple tools: {}", j.0, tools_vec.join(", "));
                            } else {
                                real_name = tools_vec[0].clone();
                                name_id = real_name.split("__").collect();
                                eprintln!("call real tool `{}` by model returned `{}`", real_name, j.0);
                            }
                        } else {
                            eprintln!("can't find real tool by model returned `{}`", j.0);
                        }
                    }
                    let (mut result, language, is_image) = match try_call_tool(
                        &name_id,
                        &j.1,
                        //j.3.clone(),
                        model,
                        Some(tool_schema.clone()),
                        Some(client.clone()),
                        Some(para_builder_for_sub_agent.clone()),
                        true, // is main agent
                    ).await {
                        Ok(inner_result) => {
                            match inner_result {
                                Ok((result, file_option)) => {
                                    try_count = 0;
                                    if name_id[0].starts_with("load_image") {
                                        (result, get_file_language(file_option), true)
                                    } else {
                                        (result, get_file_language(file_option), false)
                                    }
                                },
                                Err(e) => {
                                    try_count += 1;
                                    if is_first {
                                        if try_count >= 3 {
                                            return Err(e)
                                        } else {
                                            eprintln!("call tool {} error, try again {}\n{}\nargs: {}", name_id[0], try_count, e, j.1);
                                            //continue 'outer
                                            (format!("call tool {} error: {}", name_id[0], e), "text".to_string(), false)
                                        }
                                    } else {
                                        eprintln!("call tool {} error, send this error to LLM\n{}\nargs: {}", name_id[0], e, j.1);
                                        try_count = 0;
                                        (format!("call tool {} error: {}", name_id[0], e), "text".to_string(), false)
                                    }
                                },
                            }
                        },
                        Err(e) => {
                            eprintln!("call tool {}, raw args: {}", name_id[0], j.1);
                            return Err(e)
                        },
                    };
                    call_tool_count += 1;

                    // 1. print call tool result
                    let test_result = if is_image {
                        "load image successful".to_string()
                    } else if result.contains("```") {
                        result.clone()
                    } else {
                        format!("```{}\n{}\n```", language, result)
                    };
                    let test_result = if let Some(content) = &j.3 {
                        if content.is_empty() {
                            format!("## 📌 {} call tool\n\n---\n\n### 🛠 run tool\n{}({})\n\n### 💡 result\n{}", call_tool_count, name_id[0], &j.1, test_result)
                        } else {
                            format!("## 📌 {} call tool\n\n---\n\n{}\n\n---\n\n### 🛠 run tool\n{}({})\n\n### 💡 result\n{}", call_tool_count, content, name_id[0], &j.1, test_result)
                        }
                    } else {
                        format!("## 📌 {} call tool\n\n---\n\n### 🛠 run tool\n{}({})\n\n### 💡 result\n{}", call_tool_count, name_id[0], &j.1, test_result)
                    };
                    eprintln!("{}", test_result);

                    // 2. add result to main message history
                    let mut call_tool_vec = Vec::new();
                    if is_first {
                        call_tool_vec.push(raw_message.clone());
                    }
                    call_tool_vec.push(ChatMessage::Tool{content: ChatMessageContent::Text(result.clone()), tool_call_id: j.2.clone()});

                    // 绘图
                    if name_id[0] == "image_generation" || name_id[0] == "edit_image" {
                        result = if name_id[0] == "image_generation" {
                            format!("create image successfull: {}", result)
                        } else if name_id[0] == "edit_image" {
                            format!("edit image successfull: {}", result)
                        } else {
                            result
                        };
                    }

                    // add each tool call result to current message history
                    if is_first {
                        history_messages.push(raw_message.clone());
                        is_first = false;
                    }
                    if is_image {
                        history_messages.push(ChatMessage::Tool{
                            content: ChatMessageContent::ContentPart(vec![ChatMessageContentPart::Image(
                                ChatMessageImageContentPart {
                                    r#type: "image_url".to_string(),
                                    image_url: ImageUrlType {
                                        url: result, // Either a URL of the image or the base64 encoded image data
                                        detail: None,
                                    },
                                },
                            )]),
                            tool_call_id: j.2,
                        });
                    } else {
                        history_messages.push(ChatMessage::Tool{content: ChatMessageContent::Text(result), tool_call_id: j.2});
                    }
                }
            },
            CallToolResult::Text(test_result) => { // normal text result, not call tool
                final_result = test_result;
                break
            },
        }
        if call_tool_count > call_tool_limit {
            final_result = format!("already call tool {} times, stop this loop", call_tool_count);
            eprintln!("{}", final_result);
            break
        }
    }
    Ok((final_result, history_messages))
}

/// sub-agent
/// 不能再开启 sub-agent，避免无限递归调用，会自动把 sub_agent 工具过滤掉
/// 也不能更新 goal 的状态，会自动把 update_goal_status 工具过滤掉
pub async fn sub_agent(
    prompt: String,
    client: Client,
    mut para_builder: ChatCompletionParametersBuilder,
    model: &str,
    tool_map: HashMap<String, Vec<String>>,
) -> Result<String, MyError> {
    let mut history_messages: Vec<ChatMessage> = vec![ChatMessage::User{
        content: ChatMessageContent::Text(prompt),
        name: None,
    }];

    let mut call_tool_count = 0;
    let call_tool_limit = call_tool_count+100; // 调用工具的次数限制
    let mut try_count = 0;
    let mut is_first: bool;
    let mut real_name: String; // 要调用的工具的真实名称，含有`_uuid第一部分`后缀

    //'outer: loop {
    loop {
        // send query to LLM
        para_builder.messages(history_messages.clone());
        let parameters = para_builder.build().map_err(|e| MyError::ChatCompletionError{error: e})?;
        let answer = call_tool_not_use_stream(client.clone(), parameters).await?;
        // if answer is call tool result, continue; else break
        match answer {
            CallToolResult::CallTool((raw_message, call_tool_result)) => { // (ChatMessage, Vec<(tool name, tool args, call tool id, content)>)
                is_first = true;
                for j in call_tool_result {
                    // call tool
                    let mut name_id: Vec<&str> = j.0.split("__").collect();
                    if name_id.len() < 2 && name_id[0] != "activate_skill" {
                        if let Some(tools_vec) = tool_map.get(&j.0) {
                            if tools_vec.len() > 1 {
                                eprintln!("sub-agent call tool error, llm only return tool name prefix `{}`, but this prefix have multiple tools: {}", j.0, tools_vec.join(", "));
                            } else {
                                real_name = tools_vec[0].clone();
                                name_id = real_name.split("__").collect();
                                eprintln!("sub-agent call real tool `{}` by model returned `{}`", real_name, j.0);
                            }
                        } else {
                            eprintln!("sub-agent can't find real tool by model returned `{}`", j.0);
                        }
                    }
                    let (mut result, _, is_image) = match Box::pin(try_call_tool(
                        &name_id,
                        &j.1,
                        //j.3.clone(),
                        model,
                        None,
                        None,
                        None,
                        false, // is main agent
                    )).await {
                        Ok(inner_result) => {
                            match inner_result {
                                Ok((result, file_option)) => {
                                    try_count = 0;
                                    if name_id[0].starts_with("load_image") {
                                        (result, get_file_language(file_option), true)
                                    } else {
                                        (result, get_file_language(file_option), false)
                                    }
                                },
                                Err(e) => {
                                    try_count += 1;
                                    if is_first {
                                        if try_count >= 3 {
                                            return Err(e)
                                        } else {
                                            eprintln!("sub-agent call tool {} error, try again {}\n{}\nargs: {}", name_id[0], try_count, e, j.1);
                                            //continue 'outer
                                            (format!("sub-agent call tool {} error: {}", name_id[0], e), "text".to_string(), false)
                                        }
                                    } else {
                                        eprintln!("sub-agent call tool {} error, send this error to LLM\n{}\nargs: {}", name_id[0], e, j.1);
                                        try_count = 0;
                                        (format!("sub-agent call tool {} error: {}", name_id[0], e), "text".to_string(), false)
                                    }
                                },
                            }
                        },
                        Err(e) => {
                            eprintln!("sub-agent call tool {}, raw args: {}", name_id[0], j.1);
                            return Err(e)
                        },
                    };
                    call_tool_count += 1;

                    // 绘图
                    if name_id[0] == "image_generation" || name_id[0] == "edit_image" {
                        result = if name_id[0] == "image_generation" {
                            format!("create image successfull: {}", result)
                        } else if name_id[0] == "edit_image" {
                            format!("edit image successfull: {}", result)
                        } else {
                            result
                        };
                    }

                    // add each tool call result to current message history
                    if is_first {
                        history_messages.push(raw_message.clone());
                        is_first = false;
                    }
                    if is_image {
                        history_messages.push(ChatMessage::Tool{
                            content: ChatMessageContent::ContentPart(vec![ChatMessageContentPart::Image(
                                ChatMessageImageContentPart {
                                    r#type: "image_url".to_string(),
                                    image_url: ImageUrlType {
                                        url: result, // Either a URL of the image or the base64 encoded image data
                                        detail: None,
                                    },
                                },
                            )]),
                            tool_call_id: j.2,
                        });
                    } else {
                        history_messages.push(ChatMessage::Tool{content: ChatMessageContent::Text(result), tool_call_id: j.2});
                    }
                }
            },
            CallToolResult::Text(test_result) => { // normal text result, not call tool
                // return result to main agent
                return Ok(test_result)
            },
        }
        if call_tool_count > call_tool_limit {
            eprintln!("sub-agent already call tool {} times, stop this loop", call_tool_count);
            break
        }
    }
    Err(MyError::OtherError{info: "sub-agent error".to_string()})
}

/// 调用 sub-agent
async fn run_sub_agent(
    name_id: &[&str],
    paras: &str,
    model: &str,
    tool_schema: Option<Vec<ChatCompletionTool>>,
    mut para_builder: Option<ChatCompletionParametersBuilder>,
    client: Option<Client>,
    indirect: bool, // 是否主 agent 间接调用，比如主 agent 调用读取大文件，改为通过 sub-agent 间接调用
) -> Result<Result<(String, Option<String>), MyError>, MyError> {
    match PARAS.tools.run(name_id[1], paras) {
        Ok((prompt_tools, _)) => {
            let parts: Vec<&str> = prompt_tools.split("---srx---").collect(); // [prompt, tool1, tool2, ...]
            let tool_map = if parts.len() > 1 {
                let mut sub_agent_tools: Vec<ChatCompletionTool> = Vec::with_capacity(parts.len()-1);
                for t in &parts[1..] {
                    let tmp_name = match t.split_once("__") {
                        Some((t_name, _)) => format!("{}__", t_name),
                        None => format!("{}__", t),
                    };
                    if tmp_name == "sub_agent__" || tmp_name == "update_goal_status__" {
                        // 禁止 sub-agent 再开启新的 sub-agent
                        // 禁止 sub-agent 更新 goal 的状态
                        eprintln!("skip tool `{}` in sub-agent", tmp_name);
                        continue
                    }
                    if let Some(ref tools) = tool_schema {
                        for t in tools {
                            if t.function.name.starts_with(&tmp_name) {
                                sub_agent_tools.push(t.clone());
                                break
                            }
                        }
                    }
                }
                let tool_map = get_tool_hashmap(&sub_agent_tools);
                if let Some(ref mut p_b) = para_builder {
                    p_b.tools(sub_agent_tools);
                }
                tool_map
            } else {
                HashMap::new()
            };
            match Box::pin(sub_agent(
                parts[0].to_string(),
                client.unwrap(),
                para_builder.unwrap(),
                model,
                tool_map,
            )).await {
                Ok(sub_agent_result) => Ok(Ok((if indirect { format!("Complete by calling the sub_agent:\n{}", sub_agent_result) } else { sub_agent_result }, None))),
                Err(e) => Ok(Err(e)),
            }
        },
        Err(e) => Ok(Err(e)),
    }
}

const LARGE_FILE_PROMPT: &str = r###"The main agent is delegating this work to preserve its context window. The requested work may require reading a large file, multiple files, long logs, generated output, structured data, source code, documentation, or other context-heavy material.

Your job is to inspect the relevant material using the available tools and return a compact, evidence-backed report. Do not include unnecessary raw content. Do not solve unrelated parts of the task.

Original main agent task:
"###;

/// 调用的skill
#[derive(Deserialize)]
struct SkillParams {
    skill_name: String,
}

/// try run tool, if error not from call tool, return Err(), else return Ok(Ok()) or Ok(Err())
async fn try_call_tool(
    name_id: &[&str],
    paras: &str,
    //info: Option<String>,
    model: &str,
    tool_schema: Option<Vec<ChatCompletionTool>>,
    client: Option<Client>,
    para_builder: Option<ChatCompletionParametersBuilder>,
    is_main_agent: bool,
) -> Result<Result<(String, Option<String>), MyError>, MyError> {
    if name_id[0] == "activate_skill" {
        //let params: SkillParams = serde_json::from_str(paras).map_err(|e| MyError::SerdeJsonFromStrError{error: e})?;
        let params: SkillParams = parse_tool_args(paras, ArgFixSpec{ array_fields: None, object_fields: None })?;
        Ok(Ok((PARAS.skills.get_skill_full_content(&params.skill_name)?, None)))
    } else if name_id.len() < 2 {
        return Ok(Err(MyError::ToolNotExistError{id: name_id[0].to_string(), info: "run_tools".to_string()}))
    } else if PARAS.tools.contain_tool_id(name_id[1]) {
        // 调用工具
        match name_id[0] {
            "image_generation" => match PARAS.tools.run(name_id[1], paras) {
                Ok((image_prompt, _)) => {
                    match image_generation(image_prompt, "gpt-image-2").await {
                        Ok(image_path) => Ok(Ok((image_path, None))),
                        Err(e) => Ok(Err(e)),
                    }
                },
                Err(e) => Ok(Err(e)),
            },
            "edit_image" => match PARAS.tools.run(name_id[1], paras) {
                Ok((facial_prompt_image, _)) => {
                    let parts: Vec<&str> = facial_prompt_image.splitn(3, "---srx---").collect(); // [是否强调面部特征, prompt, 图片路径]
                    match edit_image(parts[0] == "true", parts[2].split("---srx---").map(|img| img.to_string()).collect::<Vec<String>>(), parts[1], "gpt-image-2").await {
                        Ok(image_path) => Ok(Ok((image_path, None))),
                        Err(e) => Ok(Err(e)),
                    }
                },
                Err(e) => Ok(Err(e)),
            },
            "sub_agent" => run_sub_agent(
                name_id,
                paras,
                model,
                tool_schema,
                para_builder,
                client,
                false,
            ).await,
            #[cfg(feature = "screen-capture")]
            "screen_capture" => {
                let params: ScreenCaptureParams = match parse_tool_args::<ScreenCaptureParams>(paras, ArgFixSpec{ array_fields: None, object_fields: None }) {
                    Ok(mut p) => {
                        p.path = format!("{}/screen_capture.png", PARAS.outpath); // 将输出路径固定到当前uuid输出路径
                        p
                    },
                    Err(e) => return Ok(Err(e)),
                };
                let new_run_para = match serde_json::to_string(&params) {
                    Ok(n) => n,
                    Err(e) => return Ok(Err(MyError::JsonToStringError{error: e.into()})),
                };
                Ok(PARAS.tools.run(name_id[1], &new_run_para))
            },
            "read_file" if is_main_agent => { // 读取大文件时转为调用 sub-agent
                let read_file_para: ReadFileParams = parse_tool_args(paras, ArgFixSpec{ array_fields: None, object_fields: None })?;
                // 小文件（<500k）和 md 文件可以直接读取，大文件则通过 sub_agent 读取
                let file_path = Path::new(&read_file_para.file_path);
                let ext = if let Some(ext) = file_path.extension() {
                    Some(ext.to_ascii_lowercase().to_str().unwrap().to_string())
                } else {
                    None
                };
                let metadata = file_path.metadata()?;
                if metadata.len() < 500000 || if let Some(e) = ext { e == "md" } else { false } { // 直接读取
                    Ok(PARAS.tools.run(name_id[1], paras))
                } else { // 通过 sub_agent 读取
                    eprintln!("main agent read_file by sub-agent");
                    let sub_agent_id = PARAS.tools.get_tool_id_by_name("sub_agent").unwrap();
                    run_sub_agent(
                        &["sub_agent", &sub_agent_id],
                        &format!("{{\"prompt\": \"{}read file: {}\", \"tools\": [\"{}\"]}}", LARGE_FILE_PROMPT.replace("\n", "\\n"), read_file_para.file_path.replace("\\", "\\\\"), name_id[0]), // 这里自己构建 json 字符串，换行符、双引号、文件路径的`\`都需要转义
                        model,
                        tool_schema,
                        para_builder,
                        client,
                        true,
                    ).await
                }
            },
            "read_multiple_files" if is_main_agent => { // 读取多个文件时转为调用 sub-agent
                let read_multiple_files_para: ReadMultipleFilesParams = parse_tool_args(paras, ArgFixSpec{ array_fields: Some(vec!["paths".to_string()]), object_fields: None })?;
                let sub_agent_id = PARAS.tools.get_tool_id_by_name("sub_agent").unwrap();
                eprintln!("main agent read_multiple_files by sub-agent");
                run_sub_agent(
                    &["sub_agent", &sub_agent_id],
                    &format!("{{\"prompt\": \"{}read files:\\n{}\", \"tools\": [\"{}\"]}}", LARGE_FILE_PROMPT.replace("\n", "\\n"), read_multiple_files_para.paths.join("\\n").replace("\\", "\\\\"), name_id[0]), // 这里自己构建 json 字符串，换行符、双引号、文件路径的`\`都需要转义
                    model,
                    tool_schema,
                    para_builder,
                    client,
                    true,
                ).await
            },
            "web_search" => match PARAS.tools.run(name_id[1], paras) {
                Ok((query_count, _)) => {
                    let parts: Vec<&str> = query_count.splitn(2, "---srx---").collect(); // [搜索问题, 返回数量]
                    let count = match parts[1].parse::<u8>() {
                        Ok(c) => c,
                        Err(_) => 10,
                    };
                    match deepseek_search(parts[0], count).await {
                        Ok(search_result) => Ok(Ok((search_result, None))),
                        Err(e) => Ok(Err(e)),
                    }
                },
                Err(e) => Ok(Err(e)),
            },
            "fetch_url" => match PARAS.tools.run(name_id[1], paras) {
                Ok((url, _)) => {
                    match deepseek_fetch(&url).await {
                        Ok(fetch_result) => Ok(Ok((fetch_result, None))),
                        Err(e) => Ok(Err(e)),
                    }
                },
                Err(e) => Ok(Err(e)),
            },
            _ => Ok(PARAS.tools.run(name_id[1], paras)),
        }
    } else if PARAS.mcp_servers.contain_server_id(name_id[1]) {
        Ok(PARAS.mcp_servers.run(&name_id, paras).await)
    } else {
        return Ok(Err(MyError::ToolNotExistError{id: name_id[1].to_string(), info: "run_tools".to_string()}))
    }
}

/// markdown代码块标注语言，未包含的格式后缀直接转为小写即可
const FILE_EXT_LANGUAGE: &[(&str, &str)] = &[
    ("rs", "rust"),
    ("py", "python"),
    ("js", "javascript"),
    ("ts", "typescript"),
    ("rb", "ruby"),
    ("cs", "csharp"),
    ("kt", "kotlin"),
    ("sh", "bash"),
    ("md", "markdown"),
    ("txt", "text"),
];

/// 根据文件后缀获取所属语言
fn get_file_language(file: Option<String>) -> String {
    if let Some(f) = file {
        let path = Path::new(&f);
        match path.extension() {
            Some(ext) => {
                let ext_str = ext.to_string_lossy().to_lowercase();
                FILE_EXT_LANGUAGE
                    .iter()
                    .find(|(k, _)| k == &ext_str)
                    .map(|(_, v)| v.to_string())
                    .unwrap_or_else(|| ext_str)
            }
            None => "text".to_string(),
        }
    } else {
        "text".to_string()
    }
}

/// tool名称加了`_uuid第一部分`保证不重复，但是上下文很长时可能导致模型返回的要调用的模型丢失了`_uuid第一部分`，导致调用模型失败
/// 这里把不含`_uuid第一部分`的tool名称作为key，将对应的完整tool名称作为value，存入HashMap，当`_uuid第一部分`丢失，也能确认具体要调用的模型
fn get_tool_hashmap(tools: &Vec<ChatCompletionTool>) -> HashMap<String, Vec<String>> {
    let mut out = HashMap::new();
    for t in tools {
        let real_name = t.function.name.clone();
        let name_prefix = real_name.split("__").next().unwrap().to_string();
        out.entry(name_prefix)
            .or_insert_with(Vec::new)
            .push(real_name);
    }
    out
}

// 调用 tool 时，模型返回的 tool 的 json 参数可能类型不对
// 比如 numbers 需要是 list: `{"numnbers": [1, 2, 3]}`，但模型返回的是字符串: `{"numbers": "[1, 2, 3]"}`，导致解析为结构体时失败
// 不要去修改 json 字符串，先转为 Value 再去修改指定字段

/// 指定哪些字段允许被纠正类型
/// 例如模型可能把数组字段返回成字符串：`"numbers": "[1, 2, 3]"`
/// 这时可以把 `numbers` 放进 `array_fields`，函数会只修这个字段
#[derive(Debug, Default)]
pub struct ArgFixSpec {
    /// 这些字段期望是 array，比如 Vec<T>
    pub array_fields: Option<Vec<String>>,

    /// 这些字段期望是 object，比如某个 struct/map
    pub object_fields: Option<Vec<String>>,
}

/// 解析模型返回的 tool arguments
/// 支持两类兼容修复：
/// 1. arguments 整体被包成 JSON string
/// 2. 某些白名单字段被错误地包成 string，例如 "[1,2,3]"
pub fn parse_tool_args<T>(raw: &str, fix_spec: ArgFixSpec) -> Result<T, MyError>
where
    T: DeserializeOwned,
{
    // 第一步：把模型返回的 arguments 解析成 JSON Value
    let mut value: Value = serde_json::from_str(raw).map_err(|e| MyError::SerdeJsonFromStrError{error: e})?;

    // 第二步：兼容 arguments 整体被多包了一层字符串的情况
    // 有些接口/模型可能返回：
    // "{\"file_path\":\"Cargo.toml.txt\",\"content\":\"...\"}"
    // 外层解析后是 Value::String，需要再解析一次
    if let Value::String(inner_json) = value {
        value = serde_json::from_str(&inner_json).map_err(|e| MyError::SerdeJsonFromStrError{error: e})?;
    }

    // 第三步：只修复明确声明的字段
    // 例如字段 numbers 期望是数组，但模型给了字符串：
    // {"numbers":"[1,2,3]"}
    // 修复后变成：
    // {"numbers":[1,2,3]}
    if let Value::Object(map) = &mut value {
        // 把 array 从多封装了一层的字符串中解析出来
        if let Some(array_fields) = fix_spec.array_fields {
            for field in array_fields {
                fix_string_field_to_json_type(map, &field, |v| v.is_array());
            }
        }

        // 把 object 从多封装了一层的字符串中解析出来
        if let Some(object_fields) = fix_spec.object_fields {
            for field in object_fields {
                fix_string_field_to_json_type(map, &field, |v| v.is_object());
            }
        }
    }

    // 第四步：把修复后的 JSON Value 反序列化成结构体
    // 如果字段类型仍然不匹配，这里会正常报错
    serde_json::from_value::<T>(value).map_err(|e| MyError::StructToJsonValueError{error: e})
}

/// 如果某个字段现在是 string，并且这个 string 本身又是合法 JSON，
/// 就尝试把它解析成真正的 JSON array/object
/// `accept` 用来限制只接受目标类型：
/// array 字段只接受 array，object 字段只接受 object
fn fix_string_field_to_json_type(
    map: &mut serde_json::Map<String, Value>,
    field: &str,
    accept: impl Fn(&Value) -> bool,
) {
    // 提取 field 对应的字符串
    let Some(Value::String(s)) = map.get(field) else {
        return;
    };

    // 将提取的字符串转为 Value
    let Ok(parsed) = serde_json::from_str::<Value>(s) else {
        return;
    };

    // 如果解析得到了期望的类型则原位更新
    if accept(&parsed) {
        map.insert(field.to_string(), parsed);
    }
}
