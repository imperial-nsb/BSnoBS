fn main() {
    println!("cargo:rerun-if-changed=icon.png");

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" {
        return;
    }
    // Embedding Windows resources requires Windows-native tooling (rc.exe);
    // skip if we're cross-compiling from a non-Windows host.
    if std::env::consts::OS != "windows" {
        return;
    }

    #[cfg(target_os = "windows")]
    embed_icon();
}

#[cfg(target_os = "windows")]
fn embed_icon() {
    use std::path::PathBuf;

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let ico_path = PathBuf::from(&out_dir).join("icon.ico");

    let src = image::open("icon.png").expect("icon.png");
    let sizes = [16u32, 32, 48, 64, 128, 256];
    let mut dir = ico::IconDir::new(ico::ResourceType::Icon);
    for s in sizes {
        let resized = src
            .resize_exact(s, s, image::imageops::FilterType::Lanczos3)
            .to_rgba8();
        let img = ico::IconImage::from_rgba_data(s, s, resized.into_raw());
        dir.add_entry(ico::IconDirEntry::encode(&img).expect("encode ico entry"));
    }
    let f = std::fs::File::create(&ico_path).expect("create icon.ico");
    dir.write(f).expect("write icon.ico");

    let mut res = winresource::WindowsResource::new();
    res.set_icon(ico_path.to_str().expect("ico path"));
    res.compile().expect("compile windows resource");
}
