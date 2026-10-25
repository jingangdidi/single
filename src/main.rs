use std::process::exit;

use single::{
    parse_paras::{parse_para, PARAS},
    llm::run_llm,
};

#[tokio::main]
async fn main() {
    let parsed = match parse_para().await {
        Ok(parsed) => parsed,
        Err(e) => {
            println!("{}", e);
            exit(1);
        }
    };
    if PARAS.set(parsed).is_err() {
        println!("PARAS initialized more than once");
        exit(1);
    }

    if PARAS.all_m {
        // -M: 显示所有模型
        let mut models: Vec<_> = PARAS.api.models.iter().map(|(k, v)| (k, v.1.clone())).collect();
        models.sort_unstable_by_key(|(k, _)| *k);
        for (id, m) in models {
            if *id == PARAS.api.default {
                println!("{}: {} (default)", id, m);
            } else {
                println!("{}: {}", id, m);
            }
        }
    } else if PARAS.all_p {
        // -x: 显示所有 prompt
        let mut prompt_vec: Vec<_> = PARAS.prompt.iter().map(|(k, v)| (k, v[0].clone())).collect();
        prompt_vec.sort_unstable_by_key(|(k, _)| *k);
        for (k, v) in prompt_vec {
            println!("{}: {}", k, v);
        }
    } else if PARAS.all_t {
        println!("-T select_all_tools");
        println!("-T select_all_built_in");
        println!("-T \"built_in_group_DeepSeek search\"");
        println!("-T \"built_in_group_file system\"");
        println!("-T built_in_group_run_x");
        println!("-T select_all_external");
        println!("-T select_all_mcp");
        println!("-T mcp_server_xxx");
        // -y: 显示所有工具: (uuid, name, description)
        for (_, name, _) in PARAS.tools.get_all_tools_info() {
            println!("-T {}", name);
        }
        for (_, name, description) in PARAS.mcp_servers.get_all_tools_info() {
            println!("-T {}__: {}", name, description);
        }
    } else if PARAS.all_s {
        // -z: 显示所有skill
        let (skills, _) = PARAS.skills.skill_manager.discover_skills();
        for (i, s) in skills.iter().enumerate() {
            println!("{}: -S available-skill-{}, -S skill_group_{}", s.meta.name, i, s.group);
        }
    } else {
        // 调用 llm api 进行推理
        if let Err(e) = run_llm().await {
            println!("{}", e);
            exit(1);
        }
    }
}
