#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
"""Preprocess function to filter/prepare Wikipedia docs."""

import html

import regex as re


def preprocess(article):
    # Take out HTML escaping WikiExtractor didn't clean
    for k, v in article.items():
        article[k] = html.unescape(v)

    # Filter some disambiguation pages not caught by the WikiExtractor
    if '(消歧义)' in article['title']:
        return None

    # Filter pages with the same text with title
    if article['title'] == article['text'].replace('\n', ''):
        return None

    # Take out List/Index/Outline pages (mostly links)
    if re.match(r'(.+列表)|(.+大纲)|(.+索引)',
                article['title']):
        return None
    # 可以指|可以是|指的可能是|：
    if re.match(r'[\s\S]*(可以[指是]|指的可能是)?：\n+$',
                article['text']):
        return None

    # Return doc with `id` set to `title`
    return {'id': article['title'], 'text': article['text']}
