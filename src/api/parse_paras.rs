use std::collections::HashMap;
use std::env::{current_dir, current_exe};
use std::fs::{write, read_to_string, create_dir_all};
use std::path::{Path, PathBuf};
use std::ops::Deref;

use argh::FromArgs;
use once_cell::sync::OnceCell;
use openai_dive::v1::resources::shared::ReasoningEffort;
use ron::de::from_str;
use serde::Deserialize;

use crate::{
    prompt::create_prompt,
    error::MyError,
    tools::{
        Tools,
        built_in_tools::filesystem::utils::{
            check_path,
            normalize_path,
        },
        external_tools::{
            SingleExternalTool,
            ExternalTools,
        },
        SelectedTools,
    },
    mcp::{
        McpServers,
        stdio::{
            StdIoServer,
            StdIoServers,
        },
    },
    skills::{
        Skills,
        SkillManager,
        SelectedSkills,
    },
};

pub struct GlobalParas(OnceCell<ParsedParas>);

impl GlobalParas {
    pub const fn new() -> Self {
        Self(OnceCell::new())
    }

    pub fn set(&self, value: ParsedParas) -> Result<(), ParsedParas> {
        self.0.set(value)
    }
}

impl Deref for GlobalParas {
    type Target = ParsedParas;

    fn deref(&self) -> &Self::Target {
        self.0
            .get()
            .expect("PARAS has not been initialized")
    }
}

/// 全局变量，可以修改，存储解析的命令行参数，在解析命令行参数时初始化
pub static PARAS: GlobalParas = GlobalParas::new();

#[derive(FromArgs)]
#[argh(help_triggers("-h", "--help"))] // https://github.com/google/argh/pull/106
/// server for LLM api
struct Paras {
    /// config file, contain api_key, endpoint, model name
    #[argh(option, short = 'c')]
    config: Option<String>,

    /// question prompt or file
    #[argh(option, short = 'q')]
    question: Vec<String>,

    /// select model by index, 1-based
    #[argh(option, short = 'm')]
    model: Option<usize>,

    /// temperature
    #[argh(option, short = 't')]
    temperature: Option<f32>,

    /// top p
    #[argh(option, short = 'p')]
    top_p: Option<f32>,

    /// reasoning effort, support: low, minimal, medium(default), high, xhigh, max
    #[argh(option, short = 'e')]
    effort: Option<String>,

    /// select tools
    #[argh(option, short = 'T')]
    selected_tools:  Option<String>,

    /// select prompt by index, 1-based
    #[argh(option, short = 'P')]
    selected_prompt: Option<usize>,

    /// select skills
    #[argh(option, short = 'S')]
    selected_skills: Option<String>,

    /// allowed path, used for call tools, multiple paths separated by commas, default: ./
    #[argh(option, short = 'w')]
    allowed_path: Option<String>,

    /// skills path, default: ./skills
    #[argh(option, short = 's')]
    skills: Option<String>,

    /// system one decision model url
    #[argh(option, short = 'O')]
    systemone: Option<String>,

    /// output path, default: ./chat-log
    #[argh(option, short = 'o')]
    outpath: Option<String>,
}

/// 存储解析后的命令行参数
///#[derive(Debug, Default)]
pub struct ParsedParas {
    pub api:             Api,                         // 各api的信息
    pub question:        Vec<(String, bool)>,         // (问题, 是否图片)，可以指定多次`-q`
    pub selected_model:  SelectedModel,               // 根据索引选择的模型，1-based
    pub selected_tools:  Option<SelectedTools>,       // 选择工具，逗号间隔
    pub selected_prompt: Option<String>,              // 根据索引选择prompt，1-based
    pub selected_skills: Option<SelectedSkills>,      // 选择skill，逗号间隔
    pub allowed_path:    Vec<(PathBuf, PathBuf)>,     // allowed path (absolute path (may be not exist), normalized path) for tools, multiple paths separated by commas, default: ./
    pub prompt:          HashMap<usize, [String; 2]>, // 存储prompt
    pub outpath:         String,                      // 输出结果路径，不存在则创建，已存在则删除其中的空uuid文件夹，默认./chat-log，不需要加上`/`或`\`后缀（加上了会自动去除），保存chat记录、生成的图片、音频等
    pub tools:           Tools,                       // all tools
    pub mcp_servers:     McpServers,                  // mcp servers
    pub skills:          Skills,                      // skills
    pub systemone:       Option<String>,              // system one 决策模型的 url
    pub deepseek_search_api_key: String,              // 用于 search 和 fetch
}

/// 选择的模型的信息
pub struct SelectedModel {
    pub name:        String,
    pub endpoint:    String,
    pub api_key:     String,
    pub reasoning:   bool, // whether model is chain-of-thought
    pub temperature: Option<f32>,
    pub top_p:       Option<f32>,
    pub effort:      ReasoningEffort,
}

impl SelectedModel {
    fn new(
        model: (String, String, String, bool), // (api_key, endpoint, 模型名称, 是否支持深度思考)
        temperature: Option<f32>,
        top_p: Option<f32>,
        effort: Option<String>,
    ) -> Self {
        Self {
            name:        model.2,
            endpoint:    model.1,
            api_key:     model.0,
            reasoning:   model.3,
            temperature,
            top_p,
            effort: match effort {
                Some(e) => {
                    match e.to_lowercase().as_str() {
                        "low"     => ReasoningEffort::Low,
                        "minimal" => ReasoningEffort::Minimal,
                        "medium"  => ReasoningEffort::Medium,
                        "high"    => ReasoningEffort::High,
                        "xhigh"   => ReasoningEffort::XHigh,
                        "max"     => ReasoningEffort::Max,
                        _         => ReasoningEffort::Medium,
                    }
                },
                None => ReasoningEffort::Medium,
            },
        }
    }
}

/// 解析参数
pub async fn parse_para() -> Result<ParsedParas, MyError> {
    let para: Paras = argh::from_env();
    let (api, other_para) = Api::new(para.config).await?;
    let outpath = match para.outpath {
        Some(o) => get_outpath(&o),
        None => {
            if other_para.outpath.is_empty() {
                "./chat-log".to_string()
            } else {
                get_outpath(&other_para.outpath)
            }
        },
    };
    let model = match para.model {
        Some(m) => api.get_model_by_usize(m)?,
        None => api.get_default_model()?,
    };
    let prompt = if other_para.prompt.len() == 0 {
        create_prompt() // 参数文件没指定prompt，则使用默认prompt，使用索引获取
    } else {
        other_para.prompt
    };
    let selected_prompt = match para.selected_prompt {
        Some(i) => match prompt.get(&i) {
            Some(p) => Some(p[1].clone()),
            None => None,
        },
        None => None,
    };
    let deepseek_search_api_key = api.get_deepseek_api_key();
    let out: ParsedParas = ParsedParas{
        api: api,
        question: {
            let mut questions = Vec::with_capacity(para.question.len());
            for q in para.question {
                let tmp_file = Path::new(&q);
                if tmp_file.exists() && tmp_file.is_file() {
                    if is_image(tmp_file) {
                        questions.push((q, true));
                    } else {
                        // 读取文件内容
                        questions.push((read_to_string(tmp_file)?, false));
                    }
                } else {
                    // 直接作为问题
                    questions.push((q, false));
                }
            }
            questions
        },
        selected_model: SelectedModel::new(model, para.temperature, para.top_p, para.effort),
        selected_tools: match para.selected_tools {
            Some(t) => if t == "not_select_any_tools" {
                None
            } else if t == "select_all_tools" {
                Some(SelectedTools::All)
            } else if t == "select_all_built_in" {
                Some(SelectedTools::AllBuiltIn)
            } else if t == "select_all_external" {
                Some(SelectedTools::AllExternal)
            } else if t == "select_all_mcp" {
                Some(SelectedTools::AllMcp)
            } else if let Some(group) = t.strip_prefix("built_in_group_") {
                Some(SelectedTools::Group(group.to_string()))
            } else if let Some(server_id) = t.strip_prefix("mcp_server_") {
                Some(SelectedTools::McpServer(server_id.to_string()))
            } else if t.contains(",") { // 输入框内通过`@`选择的多个工具
                Some(SelectedTools::Multiple(t.split(",").map(|s| s.to_string()).collect()))
            } else if t.contains("__") { // single mcp tool `name__id`, select by tool name and server id
                Some(SelectedTools::McpTool(t.to_string()))
            } else { // select single built-in or external tool
                Some(SelectedTools::Single(t.to_string()))
            },
            None => None,
        },
        selected_prompt,
        selected_skills: match para.selected_skills {
            Some(t) => if t == "not_select_any_skills" {
                None
            } else if t == "select_all_available_skills" {
                Some(SelectedSkills::All)
            } else if let Some(group) = t.strip_prefix("skill_group_") { // select group skills
                Some(SelectedSkills::Group(group.to_string()))
            } else if let Some(skill_index) = t.strip_prefix("available-skill-") { // select single skill
                match skill_index.parse::<usize>() {
                    Ok(i) => Some(SelectedSkills::Single(i)),
                    Err(_) => None,
                }
            } else {
                None
            },
            None => None,
        },
        allowed_path: match para.allowed_path { // allowed path (absolute path (may be not exist), normalized path) for tools, multiple paths separated by commas, default: ./
            Some(w) => get_allowed_path(&w)?,
            None => {
                if other_para.allowed_path.is_empty() {
                    let tmp_path = PathBuf::from("./");
                    let absolute_path = match current_dir() {
                        Ok(c) => c.join(tmp_path),
                        Err(e) => return Err(MyError::OtherError{info: format!("get current dir: {}", e)}),
                    };
                    let normalized_path = absolute_path.canonicalize().unwrap();
                    vec![(absolute_path, normalized_path)]
                } else {
                    get_allowed_path(&other_para.allowed_path)?
                }
            },
        },
        prompt,
        outpath: outpath.clone(), // 输出结果路径，不存在则创建，默认./chat-log，不需要加上`/`或`\`后缀（加上了会自动去除），保存chat记录、生成的图片
        tools: Tools::new(other_para.external_tools)?, // all tools
        mcp_servers: McpServers::new(other_para.mcp_servers), // mcp servers
        skills: {
            let skills_path = match para.skills {
                Some(s) => Some(PathBuf::from(&s)),
                None => if let Some(s) = other_para.skills_path {
                    Some(PathBuf::from(&s))
                } else {
                    None
                },
            };
            if let Some(p) = skills_path {
                let skill_manager = SkillManager::from_skills_dir(p.clone());
                let (available, unavailable) = if p.exists() && p.is_dir() {
                    let (available, unavailable) = skill_manager.discover_skills();
                    (available, unavailable)
                } else {
                    (Vec::new(), Vec::new())
                };
                Skills{skill_manager, available, unavailable}
            } else {
                Skills{
                    skill_manager: SkillManager::from_skills_dir(PathBuf::from("./skills")),
                    available: Vec::new(),
                    unavailable: Vec::new(),
                }
            }
        },
        systemone: if para.systemone.is_none() {
            other_para.systemone
        } else {
            para.systemone
        },
        deepseek_search_api_key,
    };
    // 输出路径不存在则创建，已存在则删除其中的空uuid文件夹
    let tmp_outpath = Path::new(&out.outpath);
    if !(tmp_outpath.exists() && tmp_outpath.is_dir()) { // 指定输出路径不存在则创建
        if let Err(err) = create_dir_all(&tmp_outpath) {
            return Err(MyError::CreateDirAllError{dir_name: out.outpath, error: err})
        }
    }
    Ok(out)
}

/// get allowed path
fn get_allowed_path(p: &str) -> Result<Vec<(PathBuf, PathBuf)>, MyError> {
    let mut allowed_path: Vec<(PathBuf, PathBuf)> = Vec::new();
    for i in p.split(",") {
        let tmp_path = Path::new(&i);
        let absolute_path = check_path(tmp_path, true)?;
        let normalized_path = normalize_path(&absolute_path);
        allowed_path.push((absolute_path, normalized_path));
    }
    Ok(allowed_path)
}

/// 去除输出路径的后缀“/”或“\”
fn get_outpath(o: &str) -> String {
    if o.ends_with("/") {
        o.strip_suffix("/").unwrap().to_string()
    } else if o.ends_with("\\") {
        o.strip_suffix("\\").unwrap().to_string()
    } else {
        o.to_string()
    }
}

/// 判断文件是否图片
fn is_image(img: &Path) -> bool {
    let ext = img
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());

    matches!(
        ext.as_deref(),
        Some(
            "jpg" | "jpeg" | "jpe" | "jfif"
            | "png" | "gif" | "bmp" | "webp"
            | "tif" | "tiff" | "svg" | "ico"
            | "heic" | "heif" | "avif" | "apng"
        )
    )
}

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    pub name:        String, // 模型名称，例如："deepseek"
    pub pricing:     String, // 价格，例如："(in: 0.002/k, out: 0.008/k)"
    pub discription: String, // 模型描述信息，例如："全面升级为 DeepSeek-V3"
    pub is_default:  bool,   // 是否将该模型作为默认模型
    pub is_cot:      bool,   // 是否是思维链模型
}

#[derive(Clone, Debug, Deserialize)]
pub struct Config {
    pub provider: String,     // 模型提供者，例如："deepseek"、"cluade"、"openai"、"gemini"、"myself"
    pub api_key:  String,     // api-key，例如："sk-xxx"
    pub endpoint: String,     // html地址，例如："https://api.deepseek.com/v1"
    pub models:   Vec<Model>, // 所有模型
}

#[derive(Debug, Deserialize)]
struct Prompt {
    name:    String, // prompt名称
    content: String, // prompt内容
}

#[derive(Deserialize)]
struct Para {
    allowed_path:      String,                  // allowed path for tools, multiple paths separated by commas, default: ./
    #[serde(default, skip_serializing_if = "Option::is_none")]
    skills_path:       Option<String>,          // skills路径
    #[serde(default, skip_serializing_if = "Option::is_none")]
    systemone:         Option<String>,          // system one url
    outpath:           String,                  // 问答结果输出路径
    model_config:      Vec<Config>,             // 模型参数
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    prompts:           Vec<Prompt>,             // prompt
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    external_tools:    Vec<SingleExternalTool>, // external tools
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    mcp_servers:       Vec<StdIoServer>,        // mcp servers, currently only support stdio, not http
}

//#[derive(Deserialize)]
struct OptherPara {
    allowed_path:   String,                      // allowed path for tools, multiple paths separated by commas, default: ./
    skills_path:    Option<String>,              // skills路径
    systemone:      Option<String>,              // system one url
    outpath:        String,                      // 问答结果输出路径
    prompt:         HashMap<usize, [String; 2]>, // key: 序号，value: [prompt名称, prompt内容]
    external_tools: ExternalTools,               // external tools
    mcp_servers:    StdIoServers,                // mcp servers, currently only support stdio, not http
}

/// 支持的所有api
#[derive(Clone, Debug)]
pub struct Api {
    pub config:  HashMap<String, Config>,                // key: 模型提供者，value: 参数文件中该模型提供者的所有模型
    pub models:  HashMap<usize, (String, String, bool)>, // key: 模型序号，value: (模型提供者, 模型名称, 是否支持深度思考)
    pub default: usize,                                  // 默认模型的序号，序号与参数文件模型顺序一致
}

/// 实现Api的方法
impl Api {
    /// 初始化模型参数
    async fn new(config_file: Option<String>) -> Result<(Self, OptherPara), MyError> {
        // 获取参数文件，先在当前路径下检查是否有config.txt，如果没有，再去程序所在路径下检查是否有config.txt，还没有则在当前路径下生成一个模板config.txt，供用户修改
        let config_file = match config_file {
            Some(c) => PathBuf::from(c),
            None => { // 没有指定参数文件
                let tmp_config = Path::new("config.txt");
                if !(tmp_config.exists() && tmp_config.is_file()) { // 当前路径下没有config.txt
                    let cfg = match current_exe() { // 检查程序所在路径下是否有config.txt
                        Ok(exe_path) => {
                            let cfg_path = exe_path.with_file_name("config.txt"); // 将路径的程序名替换为config.txt
                            if cfg_path.exists() && cfg_path.is_file() {
                                Some(cfg_path)
                            } else {
                                None
                            }
                        }
                        Err(_) => None,
                    };
                    match cfg {
                        Some(c) => c,
                        None => { // 在当前路径下生成一个示例模板，用户基于此进行修改
                            if let Err(e) = write("config.txt", include_str!("../../config_template.txt")) {
                                return Err(MyError::WriteFileError{file: "./config.txt".to_string(), error: e})
                            }
                            return Err(MyError::ParaError{para: "The config.txt is missing, please revise the generated config.txt".to_string()})
                        },
                    }
                } else { // 当前路径下有config.txt，则使用该参数文件
                    PathBuf::from("config.txt")
                }
            },
        };
        let all_para: Para = match from_str(&read_to_string(config_file)?) {
            Ok(p) => p,
            Err(e) => return Err(MyError::ParaError{para: format!("parse config file error, {:?}", e)}),
        };
        let mut config: HashMap<String, Config> = HashMap::new();
        let mut models: HashMap<usize, (String, String, bool)> = HashMap::new();
        let mut default: usize = 0; // 默认模型的序号，第1个模型序号是1，不是0，如果参数文件中没有指定默认模型，则将第1个模型作为默认模型
        let mut default_name = "".to_string(); // 默认模型的名称，指定了多个默认模型时打印报错用
        let mut idx: usize = 0; // 模型索引序号
        for c in all_para.model_config {
            if config.contains_key(&c.provider) {
                return Err(MyError::ParaError{para: format!("The config file must not contain duplicate providers: {}", c.provider)})
            }
            config.insert(c.provider.clone(), c.clone());
            for m in c.models {
                idx += 1;
                models.insert(idx, (c.provider.clone(), m.name.clone(), m.is_cot));
                if m.is_default {
                    if default == 0 {
                        default = idx;
                        default_name = m.name.clone();
                    } else {
                        return Err(MyError::ParaError{para: format!("The default model permits only one, however, both {} and {} were detected.", default_name, m.name)})
                    }
                }
            }
        }
        if default == 0 { // 参数文件没有指定默认模型
            default = 1; // 使用参数文件中第一个模型作为默认模型
        }
        // 这里i要加1，即参数文件第一个prompt是1，因为内置“保持当前对话”是-1，“无prompt”是0
        let prompt: HashMap<usize, [String; 2]> = all_para.prompts.into_iter().enumerate().map(|(i, p)| (i+1, [p.name, p.content])).collect();
        // get mcp servers
        let mcp_servers = StdIoServers::new(all_para.mcp_servers).await?;
        // get external servers
        let external_tools = ExternalTools::new(all_para.external_tools);
        Ok(
            (
                Api {
                    config,  // key: 模型提供者，value: 参数文件中该模型提供者的所有模型
                    models,  // key: 模型序号，value: (模型提供者, 模型名称, 是否支持深度思考)
                    default, // 默认模型的序号，序号与参数文件模型顺序一致
                },
                OptherPara {
                    allowed_path: all_para.allowed_path, // allowed path for tools, multiple paths separated by commas, default: ./
                    skills_path:  all_para.skills_path,  // skills路径
                    systemone:    all_para.systemone,    // system one url
                    outpath:      all_para.outpath,      // 问答结果输出路径
                    prompt,         // key: 序号，value: (prompt名称, prompt内容)
                    external_tools, // external tools
                    mcp_servers,    // mcp servers, currently only support stdio, not http
                },
            )
        )
    }

    /// 根据模型usize序号获取模型，返回(api_key, endpoint, 模型名称, 是否支持深度思考)
    pub fn get_model_by_usize(&self, num: usize) -> Result<(String, String, String, bool), MyError> {
        match self.models.get(&num) {
            Some(value) => {
                let tmp_config = self.config.get(&value.0).unwrap();
                Ok((tmp_config.api_key.clone(), tmp_config.endpoint.clone(), value.1.clone(), value.2))
            },
            None => Err(MyError::ParaError{para: format!("no such model: {}", num)}),
        }
    }

    /// 根据指定模型字符串序号，获取模型，返回(api_key, endpoint, 模型名称, 是否支持深度思考)
    pub fn get_model_by_str(&self, n: &str) -> Result<(String, String, String, bool), MyError> {
        let num = n.parse::<usize>().map_err(|e| MyError::ParseStringError{from: n.to_string(), to: "usize".to_string(), error: e})?;
        self.get_model_by_usize(num)
    }

    /// 获取默认模型，返回(api_key, endpoint, 模型名称, 是否支持深度思考)
    pub fn get_default_model(&self) -> Result<(String, String, String, bool), MyError> {
        self.get_model_by_usize(self.default)
    }

    /// 根据指定模型名称字符串，获取模型，返回(api_key, endpoint, 模型名称, 是否支持深度思考)
    pub fn get_model_by_name(&self, name: &str) -> Result<(String, String, String, bool), MyError> {
        if let Some(n) = self.models.iter().find(|(_, v)| v.1 == name).map(|(k, _)| *k) {
            self.get_model_by_usize(n)
        } else {
            self.get_default_model()
        }
    }

    /// 获取 embedding 模型
    pub fn get_embedding_modle(&self, modle_name: Option<String>) -> Option<(String, String, String)> {
        let name = modle_name.unwrap_or("embedding".to_string());
        if let Some(n) = self.models.iter().find(|(_, v)| if name == "embedding" {v.1.to_lowercase().contains("embedding")} else {v.1 == name}).map(|(k, _)| *k) {
            if let Ok(modle_info) = self.get_model_by_usize(n) { // (api_key, endpoint, 模型名称, 是否支持深度思考)
                Some((modle_info.0, modle_info.1, modle_info.2))
            } else {
                None
            }
        } else {
            None
        }
    }

    /// 获取 deepseek 的 api key
    fn get_deepseek_api_key(&self) -> String {
        let mut api_key = String::new();
        for i in ["深度求索", "deepseek", "DeepSeek"] {
            if let Some(c) = &self.config.get(i) {
                api_key = c.api_key.clone();
                break
            }
        }
        api_key
    }
}
