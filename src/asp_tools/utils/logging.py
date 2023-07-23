import logging


def get_module_logger(module_name, log_level=logging.DEBUG) -> logging.Logger:
    # ロガーオブジェクトを作成
    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)

    # ログをコンソールに出力するハンドラを作成
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # ログメッセージのフォーマットを設定
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # ハンドラをロガーに追加
    logger.addHandler(console_handler)

    return logger
