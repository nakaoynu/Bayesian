function CodeBlock(el)
  -- Convert fenced ```math blocks into real display math.
  if el.classes:includes('math') then
    return pandoc.Para({ pandoc.Math('DisplayMath', el.text) })
  end
  return nil
end
