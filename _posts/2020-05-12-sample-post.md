---
layout: post
title: Sample Post
image: "/posts/25.png"
tags: [jekyll, docs]
---
Markdown (or Textile), Liquid, HTML & CSS go in. Static sites come out ready for deployment.

#### Headings

# Heading 1

## Heading 2

### Heading 3

#### Heading 4

##### Heading 5

###### Heading 6

#### Blockquote

> No more databases, comment moderation, or pesky updates to install—just your content.

#### Unordered List

* Jekyll
    * Nested Jekyll
    * Nested Ruby
* Ruby
* Markdown
* Liquid

#### Ordered List

1. Jekyll
    1. Nested Jekyll
    2. Nested Ruby
2. Ruby
3. Markdown
4. Liquid

#### Link

This is <a href="http://example.com/" title="Title">an example</a> inline link.

#### Paragraph

Jekyll is a simple, blog-aware, static site generator. It takes a template directory containing raw text files in various formats, runs it through a converter (like Markdown) and our Liquid renderer, and spits out a complete, ready-to-publish static website suitable for serving with your favorite web server. Jekyll also happens to be the engine behind GitHub Pages, which means you can use Jekyll to host your project’s page, blog, or website from GitHub's servers for free.

### Quoting

> “Creativity is allowing yourself to make mistakes. Design is knowing which ones to keep.”

― Scott Adams
{: .cite}

#### Image
<figure class="aligncenter">
	<img src="https://images.unsplash.com/photo-1449452198679-05c7fd30f416?ixlib=rb-0.3.5&q=80&fm=jpg&crop=entropy&s=73181f1c6d56b933b30de2bfe21fdf3b" />
	<figcaption>Photo by <a href="https://unsplash.com/rmaedavis" target="_blank">Rachel Davis</a>.</figcaption>
</figure>

## Footnotes

The quick brown fox[^1] jumped over the lazy dog[^2].

[^1]: Foxes are red
[^2]: Dogs are usually not red

Footnotes are a great way to add additional contextual details when appropriate. Ghost will automatically add footnote content to the very end of your post.

## Tables

<table>
<caption>Table Caption</caption>
<thead>
<tr>
   <th>Content categories</th>
   <th>Flow content</th>
  </tr>
</thead>
 <tbody>
  <tr>
   <td>Permitted content</td>
   <td>
    An optional <code>&lt;caption&gt;</code> element;<br />
    zero or more <code>&lt;colgroup&gt;</code> elements;<br />
    an optional <code>&lt;thead&gt;</code> element;<br />
    an optional <code>&lt;tfoot&gt;</code> element;
   </td>
  </tr>
  <tr>
   <td>Tag omission</td>
   <td>None, both the start tag and the end tag are mandatory</td>
  </tr>
  <tr>
   <td>Permitted parent elements</td>
   <td>Any element that accepts flow content</td>
  </tr>
  <tr>
   <td>Normative document</td>
   <td><a href="http://www.whatwg.org/specs/web-apps/current-work/multipage/tabular-data.html#the-table-element" rel="external nofollow">HTML5, section 4.9.1</a> (<a href="http://www.w3.org/TR/REC-html40/struct/tables.html#edef-TABLE">HTML4.01, section 11.2.1</a>)</td>
  </tr>
 </tbody>
</table>

#### Default Code Block

    This is code blog.

#### Styled Code Block

    #!/usr/bin/ruby
    $LOAD_PATH << '.'
    require "support"

    class Decade
    include Week
       no_of_yrs=10
       def no_of_months
          puts Week::FIRST_DAY
          number=10*12
          puts number
       end
    end
    d1=Decade.new
    puts Week::FIRST_DAY
    Week.weeks_in_month
    Week.weeks_in_year
    d1.no_of_months
    
    def primes_finder(n):
    
    # number range to be checked
    number_range = set(range(2, n+1))

    # empty list to append discovered primes to
    primes_list = []

    while number_range:
        prime = number_range.pop()
        primes_list.append(prime)
        multiples = set(range(prime*2, n+1, prime))
        number_range.difference_update(multiples)
