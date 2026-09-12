use nanobot::providers::{base::LLMProvider, openai_compat::OpenAICompatProvider};
use std::{
    io::{Read, Write},
    net::TcpListener,
};
fn main() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut s, _) = listener.accept().unwrap();
        let mut b = [0u8; 4096];
        let n = s.read(&mut b).unwrap();
        println!(
            "{}",
            String::from_utf8_lossy(&b[..n]).lines().next().unwrap()
        );
        s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            .unwrap();
    });
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    rt.block_on(async {
        let p = OpenAICompatProvider::new("", Some(&format!("http://{addr}/v1")), Some("fixture"))
            .with_higgs_session_cache(true);
        println!(
            "result={:?}",
            p.drop_higgs_sessions("fixture", &[123]).await
        );
    });
    server.join().unwrap();
}
