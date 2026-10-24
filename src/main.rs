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

    // 调用 llm api 进行推理
    if let Err(e) = run_llm().await {
        println!("{}", e);
        exit(1);
    }
}
