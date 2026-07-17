pub(crate) fn ensure_rustls_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// iPad/Android #275: reqwest 0.13 (rustls-no-provider) 默认用 rustls-platform-verifier 校验证书,
/// 它在 Android 上**必须先用 JNI Context 初始化系统证书库**否则首次 HTTPS 握手 panic
/// ("Expect rustls-platform-verifier to be initialized")。Tauri v2 不初始化 ndk_context, 拿不到
/// JNI 句柄 → 走不通。改用**内置的 webpki (Mozilla) 根证书**做校验 (跟 reqwest 0.12 的 rustls-tls
/// 一样), 不依赖平台证书库, 全平台一致, 无需 JNI。桌面同样适用 (公网 CA 都在 Mozilla 根里)。
fn preconfigured_webpki_tls() -> rustls::ClientConfig {
    ensure_rustls_crypto_provider();
    let mut roots = rustls::RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    rustls::ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth()
}

#[allow(dead_code)]
pub(crate) fn reqwest_client() -> reqwest::Client {
    reqwest_client_builder()
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

pub(crate) fn reqwest_client_builder() -> reqwest::ClientBuilder {
    ensure_rustls_crypto_provider();
    reqwest::Client::builder().use_preconfigured_tls(preconfigured_webpki_tls())
}

pub(crate) fn reqwest_blocking_client_builder() -> reqwest::blocking::ClientBuilder {
    ensure_rustls_crypto_provider();
    reqwest::blocking::Client::builder().use_preconfigured_tls(preconfigured_webpki_tls())
}
