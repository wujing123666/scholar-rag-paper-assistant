/** getRandomValues is available on HTTP too; randomUUID requires a secure context. */
export function createRequestId(): string {
  try {
    const bytes = new Uint8Array(16);
    globalThis.crypto.getRandomValues(bytes);
    return Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("");
  } catch {
    throw new Error("浏览器无法创建安全请求编号，请检查浏览器设置后重试。");
  }
}
